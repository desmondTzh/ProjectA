from prefect import flow
import datetime

@flow(log_prints=True)
def my_flow():
    try:
    
        current_time = datetime.datetime.now()
        print(f'THE TIME IS {current_time.strftime("%H:%M:%S")}')
    except:
        print("ERROR")
