def normalizer(txt):
    return str(txt).replace('/','-').replace('Jan','01').replace('Feb','02').replace('Mar','03').replace('Apr','04')\
                .replace('May','05').replace('Jun','06').replace('Jul','07').replace('Aug','08').replace('Sep','09').replace('Oct','10')\
        .replace('Nov','11').replace('Dec','12')