import sqlite3
from app_config import config_manager
import collections


headers = config_manager.categories.keys()

conn = sqlite3.connect(':memory:')
cur = conn.cursor()


def create_table():
    global headers
    global cur
    global conn
    query = 'create table experiments (id integer primary key autoincrement, name text, {} text, {} text, {} text)'.format(*headers)
    cur.execute(query)
    conn.commit()

def insert_experiment(exp: collections.namedtuple):
    global headers
    global cur
    global conn
    print(*exp)
    query = 'insert into experiments values (?, ?, ?, ?, ?)'
    try:
        cur.execute(query, exp)
        conn.commit()
    except Exception as e:
        print('Exception in db')
        print(e)


def show_all():
    global cur
    global conn
    query = 'select * from experiments'
    cur.execute(query)
    for row in cur.fetchall():
        print(row)



if __name__ == '__main__':
    create_table()
