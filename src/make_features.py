from pathlib import Path
import pandas as pd

RAW_XLSX = Path("data/raw/online_retail_II.xlsx")
OUT_CSV = Path("data/processed/customer_features.csv")

def load_raw(xlsx_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    return pd.concat(sheets.values(), ignore_index=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # ✅ تطبیق ستون‌های دیتاست شما با نام‌های استاندارد
    df = df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    })

    # حذف مشتری‌های بدون ID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # نوع داده‌ها
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])

    # کنسلی‌ها را حذف نمی‌کنیم؛ فقط علامت می‌زنیم
    df["IsCancelled"] = df["InvoiceNo"].str.upper().str.startswith("C")

    # مبلغ هر ردیف
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
    return df

def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    # فاکتور-محور: هر InvoiceNo یک ردیف
    inv = (
        df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
          .agg(
              InvoiceDate=("InvoiceDate", "max"),
              IsCancelled=("IsCancelled", "max"),
              InvoiceTotal=("LineTotal", "sum"),
          )
    )
    reference_date = inv["InvoiceDate"].max() + pd.Timedelta(days=1)

    # خرید واقعی (غیرکنسلی + مبلغ مثبت)
    inv_buy = inv[(inv["IsCancelled"] == False) & (inv["InvoiceTotal"] > 0)].copy()

    # RFM
    last_buy = inv_buy.groupby("CustomerID")["InvoiceDate"].max()
    recency = (reference_date - last_buy).dt.days.rename("Recency")
    frequency = inv_buy.groupby("CustomerID")["InvoiceNo"].nunique().rename("Frequency")
    monetary = inv_buy.groupby("CustomerID")["InvoiceTotal"].sum().rename("Monetary")

    # ویژگی‌های زمانی
    inv_buy["DayOfWeek"] = inv_buy["InvoiceDate"].dt.dayofweek
    inv_buy["Hour"] = inv_buy["InvoiceDate"].dt.hour

    weekend_ratio = (
        inv_buy.assign(IsWeekend=inv_buy["DayOfWeek"].isin([5, 6]))
              .groupby("CustomerID")["IsWeekend"].mean()
              .rename("Weekend_Ratio")
    )

    night_shopper = (
        inv_buy.assign(IsNight=inv_buy["Hour"] >= 20)
              .groupby("CustomerID")["IsNight"].mean()
              .rename("Night_Shopper")
    )

    # تنوع سبد (خط-محور، فقط خرید واقعی)
    df_buy_lines = df[(df["IsCancelled"] == False) & (df["LineTotal"] > 0)].copy()
    basket_diversity = df_buy_lines.groupby("CustomerID")["StockCode"].nunique().rename("Basket_Diversity")

    # نرخ کنسلی (فاکتور-محور)
    total_invoices = inv.groupby("CustomerID")["InvoiceNo"].nunique()
    cancelled_invoices = inv[inv["IsCancelled"] == True].groupby("CustomerID")["InvoiceNo"].nunique()
    return_rate = (cancelled_invoices / total_invoices).fillna(0).rename("Return_Rate")

    # جدول نهایی
    features = pd.concat(
        [recency, frequency, monetary, weekend_ratio, night_shopper, basket_diversity, return_rate],
        axis=1
    ).fillna(0).reset_index()

    return features

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw(RAW_XLSX)
    df = clean(df)
    feats = build_customer_features(df)
    feats.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, "shape:", feats.shape)
    print(feats.head())

if __name__ == "__main__":
    main()
