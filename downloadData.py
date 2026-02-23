#downloadData.py - Downloads UCI Online Retail dataset

import os
import subprocess
import sys

def installUcimlrepo():
    #Install ucimlrepo if not present
    try:
        import ucimlrepo
        return True
    except ImportError:
        print("💫 Installing ucimlrepo...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
        return True

def downloadUCI():
    print("📊 Source: UCI Machine Learning Repository")
    print("🌐 URL: https://archive.ics.uci.edu/dataset/352/online+retail")
    print()
    print("✒️ Citation:")
    print("  Chen, D. (2015). Online Retail [Dataset].")
    print("  UCI Machine Learning Repository.")
    print("  https://doi.org/10.24432/C5BW33")
    print()
    
    #Install if needed
    installUcimlrepo()
    
    #Now import and download
    from ucimlrepo import fetch_ucirepo
    
    print("Fetching dataset...")
    dataset = fetch_ucirepo(id = 352)
    df = dataset.data.features
    
    #Create data directory
    os.makedirs("data", exist_ok = True)
    
    #Save
    outputPath = "data/online_retail.csv"
    df.to_csv(outputPath, index = False)
    
    print()
    print("✅ Data Successfully Downloaded!")
    print()
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Saved to: {outputPath}")
    print()
    print("🪜 Next step -> python main.py --mode quick")


if __name__ == "__main__":
    try:
        downloadUCI()
    except Exception as e:
        print(f"\nError: {e}")
        print()
        print("🖐️ Manual Download ->")
        print("1. Go to- > https://archive.ics.uci.edu/dataset/352/online+retail")
        print("2. Click 'Download (22.6 MB)'")
        print("3. Extract 'Online Retail.xlsx'")
        print("4. Open in Excel, Save As CSV")
        print("5. Save as -> data/online_retail.csv")