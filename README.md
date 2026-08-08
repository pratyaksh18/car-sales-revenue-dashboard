# Car Sales & Revenue Analysis Dashboard
 
An Excel-based dashboard analyzing car sales and revenue performance across brands, regions, dealers, and fuel types for the period **January 2026 – July 2026**.
 
## Overview
 
This project tracks 100 individual sale transactions and rolls them up into an interactive dashboard and summary tables, giving a quick read on revenue trends, top-performing brands, and regional performance.
 
**Snapshot (All Regions):**
 
| Metric | Value |
|---|---|
| Total Revenue | ₹727,952,000 |
| Total Units Sold | 763 |
| Average Order Value | ₹7,279,520 |
| Top Brand | Mahindra |
 
## File Structure
 
The workbook (`Car_Sales_Revenue_Dashboard.xlsx`) contains three sheets:
 
| Sheet | Description |
|---|---|
| **Dashboard** | Top-level KPIs (total revenue, units sold, average order value, top brand) with a region filter |
| **Sales Data** | Raw transaction-level records — the single source of truth for all other sheets |
| **Summary** | Pivoted breakdowns: monthly revenue trend, revenue by brand, by region, and by fuel type |
 
A flat CSV export of the raw data (`Car_Sales_Revenue_Dashboard.csv`) is also included for use outside Excel — e.g. in Python, SQL, or BI tools.
 
## Data Dictionary
 
Each row in **Sales Data** represents one transaction:
 
| Column | Description |
|---|---|
| `Sale ID` | Unique transaction identifier (e.g. `S0001`) |
| `Date` | Date of sale |
| `Brand` | Car manufacturer (e.g. Tata, Mahindra, Maruti Suzuki, Honda, Hyundai, Kia, Toyota) |
| `Model` | Vehicle model |
| `Fuel Type` | Petrol, Diesel, CNG, or Electric |
| `Region` | North, South, East, West, or Central |
| `Dealer` | Dealership that closed the sale |
| `Units Sold` | Number of units in the transaction |
| `Unit Price (INR)` | List price per unit, in INR |
| `Discount (%)` | Discount applied to the sale |
| `Revenue (INR)` | Net revenue for the transaction |
 
## Key Insights
 
- **Top brand by revenue:** Mahindra, driven by strong sales of higher-priced models like the Scorpio and XUV700.
- **Revenue by fuel type:** Diesel leads (₹223.1M), followed by Petrol (₹165.4M), Electric (₹143.3M), and CNG (₹196.2M).
- **Regional performance:** Central region generates the highest revenue (₹172.7M), with West and North close behind.
- **Monthly trend:** Revenue peaked in April 2026 (₹157M) before dipping sharply in June (₹20M).
## Usage
 
1. Open `Car_Sales_Revenue_Dashboard.xlsx` in Excel or Google Sheets.
2. Use the **Select Region** dropdown on the Dashboard sheet to filter KPIs by region.
3. Refer to the **Summary** sheet for pre-built breakdowns by month, brand, region, and fuel type.
4. For programmatic analysis, use `Car_Sales_Revenue_Dashboard.csv` (raw transaction data only).
## Requirements
 
- Microsoft Excel (2007+) or a compatible spreadsheet application (Google Sheets, LibreOffice Calc)
## License
 
Add your preferred license here (e.g. MIT).
