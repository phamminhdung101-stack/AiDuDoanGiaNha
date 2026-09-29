import pandas as pd
from backend.connect_db import get_connection

def run():
    df = pd.read_csv('database/dataset.csv')
    df = df.fillna('')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM houses")
    if cursor.fetchone()[0] > 0:
        print("Data already exists. Skipping import.")
        return
        
    print(f"Importing {len(df)} rows into houses table...")
    sql = """
        INSERT INTO houses (title, price, area, district, city, street, bedrooms, bathrooms, house_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    values = []
    for _, row in df.iterrows():
        try:
            price = float(row['price'])
        except:
            price = 0
        try:
            area = float(row['area'])
        except:
            area = 0
        try:
            rooms = int(row['rooms'])
        except:
            rooms = 0
        try:
            toilets = int(row['toilets'])
        except:
            toilets = 0
            
        values.append((
            str(row['title'])[:250],
            price,
            area,
            str(row['location'])[:120],
            'Hồ Chí Minh',
            str(row['address'])[:250],
            rooms,
            toilets,
            'Nhà phố' # generic default for the dataset since it's not present, or maybe 'Nhà ở'
        ))
        
    batch_size = 1000
    for i in range(0, len(values), batch_size):
        cursor.executemany(sql, values[i:i+batch_size])
        conn.commit()
        
    print("Import completed!")
    cursor.close()
    conn.close()

if __name__ == '__main__':
    run()
