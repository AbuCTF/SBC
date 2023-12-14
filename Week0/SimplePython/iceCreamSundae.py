import random

customers = ['Jimmy', 'Kim', 'John', 'Stacie'] 
winner = random.choice(customers)
flavor = 'vanilla'
print('Congratulations ' + winner +' you have won an ice cream sundae!') 
prompt = 'Would you like a cherry on top? '
wants_cherry = input(prompt)
order = flavor + ' sundae '
if (wants_cherry == 'yes'): order = order + ' with a cherry on top'
print('One ' + order + ' for ' + winner + ' coming right up...')