import pandas as pd
import MySQLdb as dbapi
from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser(description='Parse SQL Query')
    parser.add_argument('--query', nargs='?', default="select p0,p1,p2,p3,p4,p5,p6,p7,p8,p9 from nick_gianno.ncf_top_products where cookie_id=1000623420493358791",
                        help='Input SQL query.')

    return parser.parse_args()

args = parse_args()
QUERY = args.query

db=dbapi.connect(host='localhost',user='nick_gianno',passwd='2020Nick@#$2020',)
cur=db.cursor()
cur.execute(QUERY)
result=cur.fetchall()
result = pd.DataFrame(result)
print(result.loc[0])
