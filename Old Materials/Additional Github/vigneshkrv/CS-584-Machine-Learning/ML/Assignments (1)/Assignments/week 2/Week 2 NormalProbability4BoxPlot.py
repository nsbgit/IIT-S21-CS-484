from scipy.stats import norm

Normal_Q1 = norm.ppf(0.25, loc = 0, scale = 1)
Normal_Q2 = norm.ppf(0.50, loc = 0, scale = 1)
Normal_Q3 = norm.ppf(0.75, loc = 0, scale = 1)

print('Normal Q1 = ', Normal_Q1)
print('Normal Q2 = ', Normal_Q2)
print('Normal Q3 = ', Normal_Q3)

IQR = Normal_Q3 - Normal_Q1
Whisker_Lower = Normal_Q1 - 1.5 * IQR
Whisker_Upper = Normal_Q3 + 1.5 * IQR

Outside_Prob_L = norm.cdf(Whisker_Lower, loc = 0, scale = 1) # Area to the left
Outside_Prob_U = norm.sf(Whisker_Upper, loc = 0, scale = 1)  # Area to the right

print('Lower Whisker (Outside Probability) = ', Whisker_Lower, '(', Outside_Prob_L, ')')
print('Upper Whisker (Outside Probability) = ', Whisker_Upper, '(', Outside_Prob_U, ')')

