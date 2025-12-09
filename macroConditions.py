# conditions are strings given as XYZ,
# where X is H if recent output High, L otherwise
# where Y is H if current inflation High, L otherwise
# where Z is H if current interest rate is high, L otherwise
def macroCondRankWeight(metstr,cond):
    weight = 1
    if metstr == 'currentRatio':
        if cond == 'HHH':
            weight = 2
        elif cond == 'LLL':
            weight = 0
        elif cond == 'HHL':
            weight = 1
        elif cond == 'HLH':
            weight = 1
        elif cond == 'LHH':
            weight = 1.5
        elif cond == 'HLL':
            weight = 0.5
        elif cond == 'LHL':
            weight = 2
        elif cond == 'LLH':
            weight = 1

    return weight