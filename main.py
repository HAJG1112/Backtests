import indicator as indicator

def RunIndicator():
    i = indicator.Sumo('^GSPC', '2014-01-01', '2019-01-01')
    print(i.indicator_values(3, 10, 4,8,10,26))

if __name__ == "__main__":
    RunIndicator()
