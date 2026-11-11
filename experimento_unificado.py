# -*- coding: utf-8 -*-
"""
Experimento unificado - Diagnóstico de falhas no Tennessee Eastman Process (TEP)
Santos; Bernardo; Rocha; Coelho (PUC Goiás, 2026)

Protocolo único (PU), que combina o melhor de P1, P2 e P3 e corrige as limitações
apontadas no artigo:
  * divisão por execução completa (400 treino / 100 validação por classe), com manifesto salvo;
  * treino com rótulo físico (amostras pré-falha = classe normal);
  * UMA predição sobre o teste inteiro, avaliada sob as TRÊS convenções de rótulo
    (pós-falha como P1/P2, rótulo físico como P3, rótulo da execução);
  * busca de hiperparâmetros em grade, só na validação; teste consultado uma única vez;
  * várias sementes no treino final;
  * mesmo número de threads para todos os modelos (custo comparável);
  * tempo de treino, inferência, tamanho do modelo e PICO DE MEMÓRIA;
  * estatística por execução: Friedman, Wilcoxon-Holm, W de Kendall, rank-biserial,
    bootstrap por execução (IC 95% de F1 macro e MCC, e das diferenças entre pares);
  * voto majoritário por execução;
  * piloto do efeito da unidade de divisão (execução x amostra).

Uso (na pasta onde estão os 4 arquivos .RData):
    python experimento_unificado.py --etapa tudo
Etapas isoladas: dados | busca | final | estatistica | piloto | figuras
Teste rápido do código (minutos): python experimento_unificado.py --etapa tudo --rapido
"""
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--etapa", default="tudo", choices=["tudo", "dados", "busca", "final", "estatistica", "piloto", "figuras", "ablacao"])
ap.add_argument("--dados", default=".", help="pasta com os 4 arquivos .RData")
ap.add_argument("--saida", default="resultados")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--sementes", default="42,123,2024")
ap.add_argument("--passo-busca", type=int, default=5, help="na busca, usa 1 a cada N amostras do treino (o treino final usa todas)")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--rapido", action="store_true", help="usa 20 execuções por classe; só para testar o código")
ap.add_argument("--sintetico", action="store_true", help=argparse.SUPPRESS)
ap.add_argument("--modelos", default="RL,SVM,AD,RF,GB,XGB")
args = ap.parse_args()

# threads fixadas ANTES de importar numpy/sklearn: custo comparável entre modelos
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = str(args.threads)

import json, time, platform, threading, itertools, hashlib, sys
import numpy as np
import pandas as pd
import joblib
import psutil
from scipy import stats
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score,
                             accuracy_score, confusion_matrix, precision_recall_fscore_support)
try:
    from xgboost import XGBClassifier
    import xgboost
except ImportError:
    XGBClassifier = None

OUT = args.saida
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
SEEDS = [int(s) for s in args.sementes.split(",")]
MODELOS = [m.strip() for m in args.modelos.split(",")]
N_CLASSES = 21
ONSET = {"train": 20, "test": 160}  # falha introduzida após estas amostras
SPLIT_SEED = 42
N_TR, N_VA = (16, 4) if args.rapido else (400, 100)
N_TEST_RUNS = 20 if args.rapido else 500


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ------------------------------------------------------------------ DADOS
ARQ = {"ff_tr": "TEP_FaultFree_Training.RData", "f_tr": "TEP_Faulty_Training.RData",
       "ff_te": "TEP_FaultFree_Testing.RData", "f_te": "TEP_Faulty_Testing.RData"}


def ler_rdata(nome):
    import pyreadr
    caminho = os.path.join(args.dados, nome)
    log("lendo", caminho)
    res = pyreadr.read_r(caminho)
    df = next(iter(res.values()))
    return df


def sintetico(particao):
    """dados falsos com a mesma estrutura, só para testar o código"""
    rng = np.random.default_rng(0)
    n_s = 500 if particao == "train" else 960
    n_r = N_TR + N_VA if particao == "train" else N_TEST_RUNS
    rows = []
    for f in range(N_CLASSES):
        for r in range(1, n_r + 1):
            s = np.arange(1, n_s + 1)
            X = rng.normal(size=(n_s, 52)).astype(np.float32)
            if f > 0:
                X[s > ONSET[particao], f % 52] += 1.5 if f not in (3, 9, 15) else 0.05
            rows.append(np.column_stack([np.full(n_s, f), np.full(n_s, r), s, X]))
    a = np.vstack(rows).astype(np.float32)
    return a


def preparar_dados():
    """converte os .RData em arrays compactos (float32) e salva o manifesto de divisão"""
    if os.path.exists(os.path.join(CACHE, "test.npz")):
        log("cache de dados já existe; pulando leitura")
        return
    for part, (a, b) in {"train": ("ff_tr", "f_tr"), "test": ("ff_te", "f_te")}.items():
        if args.sintetico:
            arr = sintetico(part)
        else:
            blocos = []
            for k in (a, b):
                df = ler_rdata(ARQ[k])
                ids = df[["faultNumber", "simulationRun", "sample"]].to_numpy(np.float32)
                X = df.drop(columns=["faultNumber", "simulationRun", "sample"]).to_numpy(np.float32)
                assert X.shape[1] == 52, X.shape
                blocos.append(np.hstack([ids, X]))
                del df
            arr = np.vstack(blocos)
            del blocos
        f, r, s = arr[:, 0].astype(np.int16), arr[:, 1].astype(np.int16), arr[:, 2].astype(np.int16)
        X = np.ascontiguousarray(arr[:, 3:])
        del arr
        # auditoria
        aud = {"linhas": int(len(X)), "ausentes": int(np.isnan(X).sum()), "infinitos": int(np.isinf(X).sum()),
               "execucoes_por_classe": {int(c): int(len(np.unique(r[f == c]))) for c in range(N_CLASSES)}}
        if args.rapido:
            keep = r <= (N_TR + N_VA if part == "train" else N_TEST_RUNS)
            f, r, s, X = f[keep], r[keep], s[keep], X[keep]
        log(part, "auditoria:", json.dumps({k: v for k, v in aud.items() if k != "execucoes_por_classe"}))
        json.dump(aud, open(os.path.join(OUT, f"auditoria_{part}.json"), "w"), indent=1)
        np.savez(os.path.join(CACHE, f"{part}.npz"), f=f, r=r, s=s, X=X)
        del X
    # manifesto de divisão por execução (por classe, semente fixa)
    d = np.load(os.path.join(CACHE, "train.npz"))
    rng = np.random.default_rng(SPLIT_SEED)
    man = {"semente": SPLIT_SEED, "treino": {}, "validacao": {}}
    for c in range(N_CLASSES):
        runs = np.unique(d["r"][d["f"] == c])
        rng.shuffle(runs)
        man["treino"][c] = sorted(int(x) for x in runs[:N_TR])
        man["validacao"][c] = sorted(int(x) for x in runs[N_TR:N_TR + N_VA])
        assert not set(man["treino"][c]) & set(man["validacao"][c])
    txt = json.dumps(man, sort_keys=True)
    man["sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(man, open(os.path.join(OUT, "manifesto_divisao.json"), "w"), indent=1)
    log("manifesto salvo; sha256 =", man["sha256"][:16])


def carregar(part):
    d = np.load(os.path.join(CACHE, f"{part}.npz"))
    f, r, s, X = d["f"].astype(np.int64), d["r"].astype(np.int64), d["s"].astype(np.int64), d["X"]
    onset = ONSET[part]
    y_fis = np.where(s > onset, f, 0)          # rótulo físico
    pos = s > onset                             # amostras pós-início da falha
    run_id = f * 10000 + r                      # trajetória independente dentro da partição
    return dict(f=f, r=r, s=s, X=X, y_fis=y_fis, y_exec=f, pos=pos, run=run_id)


def mascaras_treino(d):
    man = json.load(open(os.path.join(OUT, "manifesto_divisao.json")))
    tr = np.zeros(len(d["f"]), bool); va = np.zeros(len(d["f"]), bool)
    for c in range(N_CLASSES):
        sel = d["f"] == c
        tr |= sel & np.isin(d["r"], man["treino"][str(c)])
        va |= sel & np.isin(d["r"], man["validacao"][str(c)])
    assert not (tr & va).any()
    return tr, va


# ------------------------------------------------------------------ MODELOS
def construir(nome, p, seed):
    T = args.threads
    if nome == "RL":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=1000, tol=1e-3))
    if nome == "SVM":
        return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=p["alpha"], max_iter=200, tol=1e-4,
                             average=True, class_weight="balanced", n_jobs=T, random_state=seed))
    if nome == "AD":
        return DecisionTreeClassifier(min_samples_leaf=p["min_samples_leaf"], random_state=seed)
    if nome == "RF":
        return RandomForestClassifier(n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"],
                                      max_features=p["max_features"], n_jobs=T, random_state=seed)
    if nome == "GB":
        return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
                                              max_iter=500, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=seed)
    if nome == "XGB":
        if XGBClassifier is None:
            raise RuntimeError("xgboost não instalado")
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=T, random_state=seed)


GRADE = {
    "RL": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "SVM": [{"alpha": a} for a in (1e-6, 1e-5, 1e-4)],
    "AD": [{"min_samples_leaf": m} for m in (5, 20, 50)],
    "RF": [{"n_estimators": 200, "min_samples_leaf": m, "max_features": mf} for m in (20, 50) for mf in ("sqrt", 0.3)],  # folha >= 20: com 5,25 M amostras, folha 5 gera RF de ~20 GB
    "GB": [{"learning_rate": lr, "max_leaf_nodes": ml} for lr in (0.05, 0.1) for ml in (31, 63)],
    "XGB": [{"n_estimators": n, "max_depth": md, "learning_rate": 0.1} for n in (200, 500) for md in (6, 8)],
}
NOMES = {"RL": "Regressão logística", "SVM": "SVM linear", "AD": "Árvore de decisão",
         "RF": "Random Forest", "GB": "Gradient boosting", "XGB": "XGBoost"}


class PicoMemoria:
    """mede o pico de memória residente do processo durante um bloco"""
    def __enter__(self):
        self.p = psutil.Process(); self.base = self.p.memory_info().rss; self.pico = self.base
        self.ativo = True; self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self
    def _loop(self):
        while self.ativo:
            self.pico = max(self.pico, self.p.memory_info().rss); time.sleep(0.05)
    def __exit__(self, *a):
        self.ativo = False; self.t.join()
    @property
    def mb(self):
        return (self.pico - self.base) / 2**20


def metricas(y, yp):
    return dict(acc=accuracy_score(y, yp), acc_bal=balanced_accuracy_score(y, yp),
                f1_macro=f1_score(y, yp, average="macro", labels=range(N_CLASSES), zero_division=0),
                mcc=matthews_corrcoef(y, yp))


# ------------------------------------------------------------------ BUSCA
def busca():
    d = carregar("train"); tr, va = mascaras_treino(d)
    sub = tr & (d["s"] % args.passo_busca == 0)
    Xtr, ytr = d["X"][sub], d["y_fis"][sub]
    Xva, yva = d["X"][va], d["y_fis"][va]
    log(f"busca: treino {len(ytr):,} amostras (1 a cada {args.passo_busca}), validação {len(yva):,}")
    linhas = []
    arq = os.path.join(OUT, "busca_hiperparametros.csv")
    feitos = set()
    if os.path.exists(arq):
        antigo = pd.read_csv(arq); linhas = antigo.to_dict("records"); feitos = {(x["modelo"], x["params"]) for x in linhas}
    for m in MODELOS:
        for p in GRADE[m]:
            chave = (m, json.dumps(p, sort_keys=True))
            if chave in feitos:
                continue
            t0 = time.perf_counter(); mdl = construir(m, p, SEEDS[0]).fit(Xtr, ytr); tt = time.perf_counter() - t0
            mt = metricas(yva, mdl.predict(Xva))
            linhas.append(dict(modelo=m, params=chave[1], treino_s=tt, **mt))
            log(f"  {m} {p}: F1 macro val = {mt['f1_macro']:.4f} ({tt:.0f} s)")
            pd.DataFrame(linhas).to_csv(arq, index=False)
    df = pd.DataFrame(linhas)
    melhor = {m: json.loads(df[df.modelo == m].sort_values("f1_macro", ascending=False).iloc[0]["params"]) for m in MODELOS}
    json.dump(melhor, open(os.path.join(OUT, "config_congelada.json"), "w"), indent=1)
    log("configurações congeladas:", melhor)


# ------------------------------------------------------------------ FINAL
def final():
    cfg = json.load(open(os.path.join(OUT, "config_congelada.json")))
    d = carregar("train"); tr, va = mascaras_treino(d)
    Xtr, ytr = d["X"][tr | va], d["y_fis"][tr | va]       # treino final = treino + validação (500 execuções)
    del d
    te = carregar("test")
    log(f"treino final: {len(ytr):,} amostras; teste: {len(te['f']):,} amostras (consultado uma única vez)")
    os.makedirs(os.path.join(OUT, "predicoes"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "modelos"), exist_ok=True)
    custos = []
    for m in MODELOS:
        for sd in SEEDS:
            arq_pred = os.path.join(OUT, "predicoes", f"{m}_s{sd}.npy")
            if os.path.exists(arq_pred):
                log(f"  {m} semente {sd}: já existe"); continue
            mdl = construir(m, cfg[m], sd)
            with PicoMemoria() as pm:
                t0 = time.perf_counter(); mdl.fit(Xtr, ytr); tt = time.perf_counter() - t0
            arq_m = os.path.join(OUT, "modelos", f"{m}_s{sd}.joblib")
            joblib.dump(mdl, arq_m, compress=0)
            tam = os.path.getsize(arq_m) / 2**20
            with PicoMemoria() as pi:
                t0 = time.perf_counter(); yp = mdl.predict(te["X"]); ti = time.perf_counter() - t0
            np.save(arq_pred, yp.astype(np.int8))
            custos.append(dict(modelo=m, semente=sd, treino_s=tt, inferencia_s=ti,
                               us_por_amostra=ti / len(yp) * 1e6, tamanho_mb=tam,
                               pico_mem_treino_mb=pm.mb, pico_mem_inferencia_mb=pi.mb))
            log(f"  {m} s{sd}: treino {tt:.0f} s; inferência {ti:.1f} s; {tam:.1f} MB; pico treino {pm.mb:.0f} MB")
            arq_c = os.path.join(OUT, "custos.csv")
            pd.DataFrame([custos[-1]]).to_csv(arq_c, mode="a", header=not os.path.exists(arq_c), index=False)
            if m in ("RF",) and sd != SEEDS[0]:
                os.remove(arq_m)  # RF é grande; guarda só a primeira semente


# ------------------------------------------------------------------ ESTATÍSTICA
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k]); adj[k] = min(1, run)
    return adj


def f1_mcc_de_cm(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean()
    t = cm.sum(1).astype(float); p = cm.sum(0).astype(float); c = tp.sum(); s = cm.sum()
    num = c * s - (t * p).sum(); d = np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return f1, (num / d if d > 0 else 0.0)


def estatistica():
    te = carregar("test")
    # pós-falha: como P1, corta as primeiras 160 amostras de TODAS as execuções (inclusive normais)
    conv = {"pos_falha": (te["y_fis"], te["s"] > ONSET["test"]),
            "rotulo_fisico": (te["y_fis"], np.ones(len(te["f"]), bool)),
            "rotulo_execucao": (te["y_exec"], np.ones(len(te["f"]), bool))}
    preds = {}
    for m in MODELOS:
        for sd in SEEDS:
            arq = os.path.join(OUT, "predicoes", f"{m}_s{sd}.npy")
            if os.path.exists(arq):
                preds[(m, sd)] = np.load(arq).astype(np.int64)
    linhas, por_classe = [], []
    for (m, sd), yp in preds.items():
        for cn, (y, mk) in conv.items():
            linhas.append(dict(modelo=m, semente=sd, convencao=cn, **metricas(y[mk], yp[mk])))
        y, mk = conv["pos_falha"]
        pr, rc, f1, _ = precision_recall_fscore_support(y[mk], yp[mk], labels=range(N_CLASSES), zero_division=0)
        for c in range(N_CLASSES):
            por_classe.append(dict(modelo=m, semente=sd, classe=c, precisao=pr[c], revocacao=rc[c], f1=f1[c]))
    met = pd.DataFrame(linhas); met.to_csv(os.path.join(OUT, "metricas_por_semente.csv"), index=False)
    resumo = met.groupby(["convencao", "modelo"])[["acc", "acc_bal", "f1_macro", "mcc"]].agg(["mean", "std"])
    resumo.to_csv(os.path.join(OUT, "metricas_resumo.csv"))
    pd.DataFrame(por_classe).groupby(["modelo", "classe"]).mean(numeric_only=True).drop(columns="semente") \
        .to_csv(os.path.join(OUT, "metricas_por_classe.csv"))
    log("\n" + met[met.convencao == "pos_falha"].groupby("modelo")[["f1_macro", "mcc", "acc_bal"]].mean()
        .sort_values("f1_macro", ascending=False).round(4).to_string())

    # predição de referência por modelo = semente 1 (para testes por execução)
    sd0 = SEEDS[0]
    ms = [m for m in MODELOS if (m, sd0) in preds]
    y, mk = conv["pos_falha"]
    runs = te["run"][mk]; u, inv = np.unique(runs, return_inverse=True)
    cls_run = np.zeros(len(u), int); cls_run[inv] = te["f"][mk]
    acc_run = {m: np.bincount(inv, weights=(preds[(m, sd0)][mk] == y[mk])) / np.bincount(inv) for m in ms}
    A = np.column_stack([acc_run[m] for m in ms])
    if len(ms) >= 3:
        chi, p = stats.friedmanchisquare(*A.T)
        W = chi / (A.shape[0] * (len(ms) - 1))
        ranks = stats.rankdata(-A, axis=1).mean(0)
        pares, ps = [], []
        for i, j in itertools.combinations(range(len(ms)), 2):
            dif = A[:, i] - A[:, j]; nz = dif[dif != 0]
            st = stats.wilcoxon(A[:, i], A[:, j], zero_method="wilcox")
            r = stats.rankdata(np.abs(nz)); rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else 0
            pares.append(dict(a=ms[i], b=ms[j], dif_media=dif.mean(), rank_biserial=rb, p=st.pvalue)); ps.append(st.pvalue)
        pp = pd.DataFrame(pares); pp["p_holm"] = holm(ps)
        pp.to_csv(os.path.join(OUT, "wilcoxon_holm.csv"), index=False)
        json.dump(dict(friedman_chi2=chi, gl=len(ms) - 1, p=p, kendall_W=W, n_execucoes=int(A.shape[0]),
                       ranking_medio=dict(zip(ms, ranks.tolist()))),
                  open(os.path.join(OUT, "friedman.json"), "w"), indent=1)
        log(f"Friedman chi2({len(ms)-1}) = {chi:.2f}; p = {p:.2e}; W = {W:.3f}")

    # bootstrap por execução, estratificado por classe (IC 95% de F1 macro e MCC, e diferenças)
    def cm_por_run(yp):
        idx = inv * (N_CLASSES * N_CLASSES) + y[mk] * N_CLASSES + yp[mk]
        return np.bincount(idx, minlength=len(u) * N_CLASSES**2).reshape(len(u), N_CLASSES, N_CLASSES)
    CM = {m: cm_por_run(preds[(m, sd0)]) for m in ms}
    rng = np.random.default_rng(2026)
    grupos = [np.where(cls_run == c)[0] for c in range(N_CLASSES)]
    B = args.bootstrap
    bs = {m: np.zeros((B, 2)) for m in ms}
    for b in range(B):
        amostra = np.concatenate([rng.choice(g, len(g), replace=True) for g in grupos])
        cnt = np.bincount(amostra, minlength=len(u))
        for m in ms:
            bs[m][b] = f1_mcc_de_cm(np.tensordot(cnt, CM[m], axes=1))
    ic = []
    for m in ms:
        f1o, mcco = f1_mcc_de_cm(CM[m].sum(0))
        lo, hi = np.percentile(bs[m], [2.5, 97.5], axis=0)
        ic.append(dict(modelo=m, f1_macro=f1o, f1_ic_inf=lo[0], f1_ic_sup=hi[0], mcc=mcco, mcc_ic_inf=lo[1], mcc_ic_sup=hi[1]))
    pd.DataFrame(ic).to_csv(os.path.join(OUT, "bootstrap_ic.csv"), index=False)
    dif = []
    for a, b in itertools.combinations(ms, 2):
        dd = bs[a] - bs[b]; lo, hi = np.percentile(dd, [2.5, 97.5], axis=0)
        dif.append(dict(a=a, b=b, dif_f1=dd[:, 0].mean(), f1_ic_inf=lo[0], f1_ic_sup=hi[0],
                        dif_mcc=dd[:, 1].mean(), mcc_ic_inf=lo[1], mcc_ic_sup=hi[1]))
    pd.DataFrame(dif).to_csv(os.path.join(OUT, "bootstrap_diferencas.csv"), index=False)

    # voto majoritário por execução (convenção pós-falha)
    voto = []
    for m in ms:
        yp = preds[(m, sd0)][mk]
        maj = np.array([np.bincount(yp[inv == k], minlength=N_CLASSES).argmax() for k in range(len(u))])
        voto.append(dict(modelo=m, f1_macro_execucao=f1_score(cls_run, maj, average="macro"),
                         acc_execucao=accuracy_score(cls_run, maj)))
    pd.DataFrame(voto).to_csv(os.path.join(OUT, "voto_majoritario.csv"), index=False)
    # matrizes de confusão normalizadas
    for m in ms:
        cm = confusion_matrix(y[mk], preds[(m, sd0)][mk], labels=range(N_CLASSES), normalize="true")
        np.savetxt(os.path.join(OUT, f"matriz_confusao_{m}.csv"), cm, delimiter=";", fmt="%.4f")
    log("estatística concluída")


# ------------------------------------------------------------------ PILOTO
def piloto(n_runs=25, repeticoes=10):
    """efeito da unidade de divisão: por execução x por amostra, mesmo volume"""
    d = carregar("train")
    res = []
    for rep in range(repeticoes):
        rng = np.random.default_rng(1000 + rep)
        sel = np.zeros(len(d["f"]), bool); grupo = np.zeros(len(d["f"]), int)
        for c in range(N_CLASSES):
            runs = rng.choice(np.unique(d["r"][d["f"] == c]), n_runs, replace=False)
            for k, r in enumerate(runs):
                mm = (d["f"] == c) & (d["r"] == r); sel |= mm; grupo[mm] = int(k < int(0.8 * n_runs))
        X, y, g = d["X"][sel], d["y_fis"][sel], grupo[sel]
        tr_e, te_e = g == 1, g == 0
        perm = rng.permutation(len(y)); tr_a = np.zeros(len(y), bool); tr_a[perm[:tr_e.sum()]] = True
        for nome, mdl in [("AD folha>=50", DecisionTreeClassifier(min_samples_leaf=50, random_state=rep)),
                          ("AD sem poda", DecisionTreeClassifier(random_state=rep)),
                          ("RF50 folha>=50", RandomForestClassifier(50, min_samples_leaf=50, n_jobs=args.threads, random_state=rep)),
                          ("RF50 sem poda", RandomForestClassifier(50, n_jobs=args.threads, random_state=rep))]:
            a = f1_score(y[te_e], mdl.fit(X[tr_e], y[tr_e]).predict(X[te_e]), average="macro")
            b = f1_score(y[~tr_a], mdl.fit(X[tr_a], y[tr_a]).predict(X[~tr_a]), average="macro")
            res.append(dict(rep=rep, modelo=nome, f1_execucao=a, f1_amostra=b, delta=b - a))
        log(f"  piloto repetição {rep + 1}/{repeticoes}")
    df = pd.DataFrame(res); df.to_csv(os.path.join(OUT, "piloto_divisao.csv"), index=False)
    s = df.groupby("modelo")["delta"].agg(["mean", "std", "count"])
    s["ic_inf"] = s["mean"] - stats.t.ppf(0.975, s["count"] - 1) * s["std"] / np.sqrt(s["count"])
    s["ic_sup"] = s["mean"] + stats.t.ppf(0.975, s["count"] - 1) * s["std"] / np.sqrt(s["count"])
    s.to_csv(os.path.join(OUT, "piloto_resumo.csv")); log("\n" + s.round(4).to_string())


# ------------------------------------------------------------------ ABLAÇÃO
def ablacao():
    """muda UM fator por vez em relação ao PU (mesmos dados, divisão, equipamento e threads):
       RF: PU + max_depth=30 (como P1)   |   GB: PU sem parada antecipada, 200 iterações (como P1/P3)"""
    cfg = json.load(open(os.path.join(OUT, "config_congelada.json")))
    d = carregar("train"); tr, va = mascaras_treino(d)
    Xtr, ytr = d["X"][tr | va], d["y_fis"][tr | va]; del d
    te = carregar("test"); mk = te["s"] > ONSET["test"]; y = te["y_fis"][mk]
    arq = os.path.join(OUT, "ablacao.csv")
    feitos = set()
    if os.path.exists(arq):
        feitos = {(r["variante"], int(r["semente"])) for r in pd.read_csv(arq).to_dict("records")}
    variantes = []
    p = cfg["RF"]
    variantes += [("RF_profundidade30", sd, lambda sd=sd, p=p: RandomForestClassifier(
        n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
        max_depth=30, n_jobs=args.threads, random_state=sd)) for sd in SEEDS[:1]]
    g = cfg["GB"]
    variantes += [("GB_sem_parada_antecipada", sd, lambda sd=sd, g=g: HistGradientBoostingClassifier(
        learning_rate=g["learning_rate"], max_leaf_nodes=g["max_leaf_nodes"], max_iter=200,
        early_stopping=False, random_state=sd)) for sd in SEEDS]
    for nome, sd, fab in variantes:
        if (nome, sd) in feitos:
            log(f"  {nome} s{sd}: já existe"); continue
        mdl = fab()
        t0 = time.perf_counter(); mdl.fit(Xtr, ytr); tt = time.perf_counter() - t0
        t0 = time.perf_counter(); yp = mdl.predict(te["X"]); ti = time.perf_counter() - t0
        mt = metricas(y, yp[mk])
        linha = dict(variante=nome, semente=sd, treino_s=tt, us_por_amostra=ti / len(yp) * 1e6, **mt)
        pd.DataFrame([linha]).to_csv(arq, mode="a", header=not os.path.exists(arq), index=False)
        log(f"  {nome} s{sd}: F1 macro = {mt['f1_macro']:.4f}; MCC = {mt['mcc']:.4f} ({tt:.0f} s)")
    log("ablação concluída")

# ------------------------------------------------------------------ FIGURAS
def figuras():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})
    pc = pd.read_csv(os.path.join(OUT, "metricas_por_classe.csv"))
    tab = pc.pivot(index="classe", columns="modelo", values="f1")
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=300)
    tab[[m for m in MODELOS if m in tab]].plot(kind="bar", ax=ax, width=0.85)
    ax.set_xlabel("Classe (0 = normal)"); ax.set_ylabel("F1 (pós-falha)"); ax.legend(ncol=6, fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_f1_por_classe.png")); plt.close()
    if os.path.exists(os.path.join(OUT, "custos.csv")):
        c = pd.read_csv(os.path.join(OUT, "custos.csv")).groupby("modelo").mean(numeric_only=True)
        m = pd.read_csv(os.path.join(OUT, "metricas_por_semente.csv"))
        m = m[m.convencao == "pos_falha"].groupby("modelo")["f1_macro"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
        for k in c.index:
            ax.scatter(c.loc[k, "us_por_amostra"], m[k]); ax.annotate(k, (c.loc[k, "us_por_amostra"], m[k]), xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("Inferência por amostra (µs)"); ax.set_ylabel("F1 macro")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pareto.png")); plt.close()
    log("figuras salvas")


def ambiente():
    info = dict(python=sys.version.split()[0], plataforma=platform.platform(), processador=platform.processor(),
                nucleos_fisicos=psutil.cpu_count(logical=False), nucleos_logicos=psutil.cpu_count(),
                ram_gb=round(psutil.virtual_memory().total / 2**30, 2), threads=args.threads,
                numpy=np.__version__, pandas=pd.__version__, sklearn=__import__("sklearn").__version__,
                xgboost=(xgboost.__version__ if XGBClassifier else None), sementes=SEEDS)
    json.dump(info, open(os.path.join(OUT, "ambiente.json"), "w"), indent=1)
    log("ambiente:", info)


if __name__ == "__main__":
    ambiente()
    e = args.etapa
    if e in ("tudo", "dados"): preparar_dados()
    if e in ("tudo", "busca"): busca()
    if e in ("tudo", "final"): final()
    if e in ("tudo", "estatistica"): estatistica()
    if e in ("tudo", "piloto"): piloto(n_runs=8 if args.rapido else 25, repeticoes=3 if args.rapido else 10)
    if e in ("tudo", "figuras"): figuras()
    if e == "ablacao": ablacao()
    log("fim")
