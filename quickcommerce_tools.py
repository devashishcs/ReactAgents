from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Load the first sheet from Excel file
df = pd.read_excel("imp_docs/quickcommerce.xlsx", sheet_name=None)  # 0 means the first sheet

@tool
def get_columns() -> list:
    """Return a list of column names from  Excel sheet as JSON format."""
    return {sheet: list(sheet_df.columns) for sheet, sheet_df in df.items()}

@tool
def group_by_po_number() -> dict:
    """
    Groups the data by 'PoNumber' (flexible column naming like 'PO No.') and returns it in JSON format.
    """
    possible_names = ["PoNumber", "PO Number", "PO No.", "PO_NO", "PO no", "po_number"]
    result = {}
    for sheet_name, sheet_df in df.items():
        found_po_column = next(
            (col for col in sheet_df.columns if col.strip() in possible_names), None
        )

        if not found_po_column:
            result[sheet_name] = "❌ No PO column found"
            continue

        grouped = sheet_df.groupby(found_po_column).apply(
            lambda x: x.to_dict(orient="records")
        )
        result[sheet_name] = grouped.to_dict()

    return result

@tool
def group_by_sku_code() -> dict:
    """
    Groups the data by 'SkuCode' (flexible column naming like 'SKU Code', 'Sku Code', etc.) and returns it in JSON format.
    """
    possible_names = ["SkuCode", "SKU"]
    result = {}
    for sheet_name, sheet_df in df.items():
        found_sku_column = next(
            (col for col in sheet_df.columns if col.strip() in possible_names), None
        )

        if not found_sku_column:
            result[sheet_name] = "❌ No SKU column found"
            continue

        grouped = sheet_df.groupby(found_sku_column).apply(
            lambda x: x.to_dict(orient="records")
        )
        result[sheet_name] = grouped.to_dict()

    return result

@tool
def group_by_city_code() -> dict:
    """
    Groups the data by 'City' (flexible column naming like 'City Code', 'CityCode', etc.) and returns it in JSON format.
    """
    possible_names = ["City", "Del Location"]
    result = {}
    for sheet_name, sheet_df in df.items():
        found_city_column = next(
            (col for col in sheet_df.columns if col.strip() in possible_names), None
        )

        if not found_city_column:
            result[sheet_name] = "❌ No City column found"
            continue

        grouped = sheet_df.groupby(found_city_column).apply(
            lambda x: x.to_dict(orient="records")
        )
        result[sheet_name] = grouped.to_dict()

    return result

@tool
def total_sales_by_sku() -> list[dict]:
    """
    Groups data by 'SkuCode' and returns total sales per SKU as a list of dictionaries.
    Handles multiple sheets with different column names (SkuCode/SKU, PoLineValueWithTax/Total Amount).
    """
    all_results = {}
    
    for sheet_name, sheet_df in df.items():
        # Determine column names for this sheet
        sku_col = "SkuCode" if "SkuCode" in sheet_df.columns else "SKU"
        amount_col = "PoLineValueWithTax" if "PoLineValueWithTax" in sheet_df.columns else "Total Amount"
        
        # Group by SKU for this sheet
        grouped = sheet_df.groupby(sku_col)[amount_col].sum().reset_index()
        
        # Add sheet results to overall results
        for _, row in grouped.iterrows():
            sku_code = row[sku_col]
            value = row[amount_col]
            
            if sku_code in all_results:
                all_results[sku_code] += value
            else:
                all_results[sku_code] = value
    
    # Convert to list of dictionaries format
    result_list = [{"SKU": sku, "AMOUNT": total} 
                   for sku, total in all_results.items()]
    return result_list

@tool
def total_sales_by_po() -> list[dict]:
    """
    Groups data by 'PoNumber' and returns total sales per PoNumber as a list of dictionaries.
    Handles multiple sheets with different column names (PO No./PoNumber, PoLineValueWithTax/Total Amount).
    """
    all_results = {}
    
    for sheet_name, sheet_df in df.items():
        # Determine column names for this sheet
        po_col = "PoNumber" if "PoNumber" in sheet_df.columns else "PO No."
        amount_col = "PoLineValueWithTax" if "PoLineValueWithTax" in sheet_df.columns else "Total Amount"
        
        # Group by SKU for this sheet
        grouped = sheet_df.groupby(po_col)[amount_col].sum().reset_index()
        
        # Add sheet results to overall results
        for _, row in grouped.iterrows():
            po_code = row[po_col]
            value = row[amount_col]
            
            if po_code in all_results:
                all_results[po_code] += value
            else:
                all_results[po_code] = value
    
    # Convert to list of dictionaries format
    result_list = [{"PO": po, "AMOUNT": total} 
                   for po, total in all_results.items()]
    return result_list


@tool
def plot_sales_by_sku(data: str) -> str:
    """
    Plots a bar graph from a JSON string representing a list of dicts with 'SKU' and 'AMOUNT'.
    Returns confirmation message.
    """
    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError:
        return "Invalid JSON input. Please pass a valid JSON string."

    if not parsed_data:
        return "No data provided to plot."

    sku_codes = [str(d["SKU"]) for d in parsed_data]
    sales = [d["AMOUNT"] for d in parsed_data]

    plt.figure(figsize=(12, 8))
    plt.bar(sku_codes, sales, color="skyblue", alpha=0.8, edgecolor='navy', linewidth=0.5)
    plt.xlabel("SKU Code", fontsize=12, fontweight='bold')
    plt.ylabel("Total Sales (AMOUNT)", fontsize=12, fontweight='bold')
    plt.title("Total Sales by SKU Code", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sales):
        plt.text(i, v + max(sales) * 0.01, f'₹{v:,.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("sales_by_sku.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return f"✅ Bar graph saved as 'sales_by_sku.png' with {len(parsed_data)} SKUs plotted"
