from datetime import date,datetime

print(date.today())

def get_paths(tdate:datetime=None):
    if tdate == None:
        tdate = date.today()
    
    val_date = "".join((str(tdate.year), "-", str(tdate.month-1)))
    tr_date = "".join((str(tdate.year), "-", str(tdate.month-2)))

    train_path = f'./data/fhv_tripdata_{tr_date}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date}.parquet'

    print(tdate)
    print(val_date)
    print(tr_date)

    print(train_path)
    print(val_path)
    return train_path, val_path

print(get_paths(2021-10-15))