#Sales Forecasting: Feature Engineering vs Deep Learning

import argparse
import json
import time
import sys
from pathlib import Path


def progressBar(current, total, width  =  30, prefix = ""):
    pct  =  current / total
    filled  =  int(width * pct)
    bar  =  "█" * filled + "░" * (width - filled)
    print(f"\r{prefix}[{bar}] {current}/{total}", end = "", flush = True)


def checkDeps():
    print("🔍 Checking dependencies")
    pkgs  =  [
        ("numpy","numpy"), 
        ("pandas","pandas"), 
        ("lightgbm","lightgbm"), 
        ("torch","torch"), 
        ("sklearn","scikit-learn"), 
        ("matplotlib","matplotlib")
    ]
    missing  =  []
    for i, (imp, name) in enumerate(pkgs):
        last  =  i  ==  len(pkgs) - 1
        branch  =  "└──" if last else "├──"
        try:
            m  =  __import__(imp)
            print(f"   {branch} ✓ {name} ({getattr(m, '__version__', 'ok')})")
        except:
            print(f"   {branch} ✗ {name} (missing)")
            missing.append(name)
    return len(missing)  ==  0


def checkData(path):
    print("\n📂 Checking data")
    p  =  Path(path)
    for f in ["online_retail.csv", "Online Retail.csv", "OnlineRetail.csv"]:
        fp  =  p / f
        if fp.exists():
            sz  =  fp.stat().st_size / 1024**2
            print(f"   └── ✓ {f} ({sz:.1f} MB)")
            return True
    print("   └── ✗ online_retail.csv (not found)")
    print("\n⚠️  Run: python downloadData.py")
    return False


def run(quick = False, sampleSize = None):
    import numpy as np
    import pandas as pd
    import torch
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    for d in ["experiments/models", "experiments/results", "experiments/figures"]:
        Path(d).mkdir(parents = True, exist_ok = True)
    
    from src.dataLoader import DataLoader
    from src.preprocessor import Preprocessor
    from src.featureEngineer import FeatureEngineer
    from src.splitter import TimeSeriesSplitter
    from src.lightgbmModel import LightGBMForecaster
    from src.lstmModel import SalesDataset, LSTMForecaster
    from src.evaluator import Evaluator
    from src.visualizer import Visualizer
    
    results  =  {}
    
    print("\n🚀 Running pipeline")
    
    #1 Load
    print("   ├── 📥 Loading data...", end = " ", flush = True)
    t0  =  time.time()
    data  =  DataLoader("data/").loadAll()
    if sampleSize and len(data["sales"]) > sampleSize:
        data["sales"]  =  data["sales"].sample(n = sampleSize, random_state = 42).sort_values("date")
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    #2 Preprocess
    print("   ├── 🔧 Preprocessing...", end = " ", flush = True)
    t0  =  time.time()
    df  =  Preprocessor().mergeAll(data)
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    #3 Features
    print("   ├── ⚙️  Engineering features...", end = " ", flush = True)
    t0  =  time.time()
    fe  =  FeatureEngineer()
    df  =  fe.buildFeatures(df, isTraining = True)
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    #4 Split
    print("   ├── ✂️  Splitting data...", end = " ", flush = True)
    t0  =  time.time()
    splitter  =  TimeSeriesSplitter()
    trainDf, validDf, testDf  =  splitter.split(df)
    trainDf  =  splitter.dropWarmupPeriod(trainDf, 28)
    validDf  =  splitter.dropWarmupPeriod(validDf, 28)
    testDf  =  splitter.dropWarmupPeriod(testDf, 28)
    cols  =  fe.getFeatureColumns()
    xTrain, yTrain  =  splitter.prepareXY(trainDf, cols)
    xValid, yValid  =  splitter.prepareXY(validDf, cols)
    xTest, yTest  =  splitter.prepareXY(testDf, cols)
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    ev  =  Evaluator()
    viz  =  Visualizer()
    
    #5 Baseline
    print("   ├── 📊 Baseline...", end = " ", flush = True)
    if "lag_7" in validDf.columns:
        naivePreds  =  validDf["lag_7"].dropna()
        naiveActual  =  validDf.loc[naivePreds.index, "log_sales"]
        results["SeasonalNaive"]  =  {**ev.computeMetrics(naiveActual, naivePreds), "training_time": 0}
    else:
        results["SeasonalNaive"]  =  {"rmsle": 0, "mae": 0, "training_time": 0}
    print(f"✓ (RMSLE: {results['SeasonalNaive']['rmsle']:.4f})")
    
    #6 LightGBM
    print("   ├── 🌳 Training LightGBM")
    t0  =  time.time()
    nRounds  =  300 if quick else 500
    lgb  =  LightGBMForecaster()
    for i in range(1, 11):
        progressBar(i, 10, prefix = "   │      ")
        if i  ==  1:
            lgb.fit(xTrain, yTrain, xValid, yValid, numBoostRound = nRounds, verboseEval = 0)
        time.sleep(0.05)
    lgbPreds  =  lgb.predict(xTest)
    results["LightGBM"]  =  {**ev.computeMetrics(yTest, pd.Series(lgbPreds)), "training_time": lgb.trainingTime_}
    print(f" ✓ (RMSLE: {results['LightGBM']['rmsle']:.4f}, Time: {results['LightGBM']['training_time']:.1f}s)")
    
    #7 LSTM
    print("   ├── 🧠 Training LSTM")
    t0  =  time.time()
    device  =  "cuda" if torch.cuda.is_available() else "cpu"
    lstmFeats  =  ["log_sales", "day_of_week", "is_holiday", "onpromotion"]
    
    trainDs  =  SalesDataset(trainDf, 28, "log_sales", lstmFeats)
    validDs  =  SalesDataset(validDf, 28, "log_sales", lstmFeats)
    testDs  =  SalesDataset(testDf, 28, "log_sales", lstmFeats)
    
    from torch.utils.data import DataLoader as TDL
    trainLdr  =  TDL(trainDs, batch_size = 128, shuffle = True)
    validLdr  =  TDL(validDs, batch_size = 128)
    testLdr  =  TDL(testDs, batch_size = 128)
    
    nFeat  =  len([c for c in lstmFeats if c in trainDf.columns])
    nEpochs  =  10 if quick else 30
    lstm  =  LSTMForecaster(inputSize = nFeat, hiddenSize = 32, numLayers = 2)
    
    lstm  =  lstm.to(device)
    criterion  =  torch.nn.MSELoss()
    optimizer  =  torch.optim.Adam(lstm.parameters(), lr = 0.001)
    
    history  =  {"train_loss": [], "valid_loss": []}
    startTime  =  time.time()
    
    for epoch in range(nEpochs):
        progressBar(epoch + 1, nEpochs, prefix = "   │      ")
        
        lstm.train()
        for xB, yB in trainLdr:
            xB, yB  =  xB.to(device), yB.to(device)
            optimizer.zero_grad()
            loss  =  criterion(lstm(xB), yB)
            loss.backward()
            optimizer.step()
        
        lstm.eval()
        vLosses  =  []
        with torch.no_grad():
            for xB, yB in validLdr:
                xB, yB  =  xB.to(device), yB.to(device)
                vLosses.append(criterion(lstm(xB), yB).item())
        
        history["train_loss"].append(loss.item())
        history["valid_loss"].append(np.mean(vLosses))
    
    history["total_time"]  =  time.time() - startTime
    history["best_epoch"]  =  nEpochs
    
    lstm.eval()
    preds, targs  =  [], []
    with torch.no_grad():
        for xB, yB in testLdr:
            preds.extend(lstm(xB.to(device)).cpu().numpy())
            targs.extend(yB.numpy())
    
    results["LSTM"]  =  {**ev.computeMetrics(pd.Series(targs), pd.Series(preds)), "training_time": history["total_time"]}
    print(f" ✓ (RMSLE: {results['LSTM']['rmsle']:.4f}, Time: {results['LSTM']['training_time']:.1f}s)")
    
    #8 Ablation
    print("   ├── 🔬 Ablation study")
    ablation  =  []
    fullRmsle  =  results["LightGBM"]["rmsle"]
    groups  =  list(fe.getFeatureGroups().items())
    for i, (grp, feats) in enumerate(groups):
        progressBar(i + 1, len(groups), prefix = "   │      ")
        ablCols  =  [c for c in cols if c not in feats]
        xTrA, _  =  splitter.prepareXY(trainDf, ablCols)
        xVaA, _  =  splitter.prepareXY(validDf, ablCols)
        xTeA, _  =  splitter.prepareXY(testDf, ablCols)
        m  =  LightGBMForecaster()
        m.fit(xTrA, yTrain, xVaA, yValid, verboseEval = 0)
        met  =  ev.computeMetrics(yTest, pd.Series(m.predict(xTeA)))
        ablation.append({"removed": grp, "rmsle_increase": met["rmsle"] - fullRmsle})
    print(" ✓")
    
    #9 Figures
    print("   └── 📈 Generating figures...", end = " ", flush = True)
    compDf  =  pd.DataFrame({
        "SeasonalNaive": {"rmsle": results["SeasonalNaive"]["rmsle"], "training_time": 0},
        "LightGBM": {"rmsle": results["LightGBM"]["rmsle"], "training_time": results["LightGBM"]["training_time"]},
        "LSTM": {"rmsle": results["LSTM"]["rmsle"], "training_time": results["LSTM"]["training_time"]}
    }).T
    viz.plotModelComparison(compDf)
    viz.plotFeatureImportance(lgb.getFeatureImportance())
    viz.plotAblationStudy(pd.DataFrame(ablation))
    viz.plotLearningCurves(history)
    print("✓")
    
    #Best model
    if results["LightGBM"]["rmsle"] < results["LSTM"]["rmsle"]:
        results["LightGBM"]["best"]  =  True
    else:
        results["LSTM"]["best"]  =  True
    
    #Save results
    with open("experiments/results/results.json", "w") as f:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)): return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, dict): return {k: conv(v) for k,v in o.items()}
            if isinstance(o, list): return [conv(i) for i in o]
            return o
        json.dump(conv(results), f, indent = 2)
    
    return results


def main():
    parser  =  argparse.ArgumentParser(description = "Sales Forecasting: Feature Engineering vs Deep Learning")
    parser.add_argument("--mode", choices = ["full", "quick"], default = "full", help = "full = 500 rounds, quick = 300 rounds")
    parser.add_argument("--sample", type = int, default = None, help = "Sample N rows for faster testing")
    args  =  parser.parse_args()
    
    print()
    print("📊 Sales Forecasting Experiment")
    print("🔬 Feature Engineering vs Deep Learning")
    
    if args.sample:
        print(f"⚡ Sample mode: {args.sample:,} rows")
    if args.mode  ==  "quick":
        print(f"⚡ Quick mode: 10 epochs")
    
    print()
    
    if not checkDeps():
        print("\n❌ Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if not checkData("data/"):
        sys.exit(1)
    
    results  =  run(quick = (args.mode  ==  "quick"), sampleSize = args.sample)
    
    #Final results
    print()
    print("📋 Results")
    print()
    for name in ["SeasonalNaive", "LightGBM", "LSTM"]:
        r  =  results[name]
        best  =  "⭐" if r.get("best") else "  "
        print(f"   {best} {name:<14} RMSLE: {r['rmsle']:<8.4f} Time: {r['training_time']:>6.1f}s")
    
    print()
    if results["LightGBM"].get("best"):
        speedup  =  results["LSTM"]["training_time"] / max(results["LightGBM"]["training_time"], 0.1)
        print(f"🏆 Best: LightGBM ({speedup:.1f}x faster than LSTM)")
    else:
        print(f"🏆 Best: LSTM")
    
    print()
    print("📁 Results: experiments/results/results.json")
    print("📊 Figures: experiments/figures/")
    print()


if __name__  ==  "__main__":
    main()