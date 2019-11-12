import indicator_error as indicator_error

if __name__ == "__main__":
    s = indicator_error.Sumo('^GSPC', '2014-01-01', '2019-01-01')
    s.get_prices_X()