import argparse

parser = argparse.ArgumentParser(description='Name and school name') # name
#parser.add_argument('integers', type=str, help='Incoming number') # single
#parser.add_argument('integers_many', type=str, nargs='+', help='Incoming number')
#parser.add_argument('sum', type=int, nargs='+', help='for sum')     # nargs : '*' 参数可以是零个或多个(Parameters can be zero or more) '+': 参数可以设置一个或多个(Parameters can be one or more) '?':参数可以设置零个或一个(Parameters can be one or two)
parser.add_argument('--family', type=str, default='nam ', help='first name')
parser.add_argument('--name', type=str, default='jiwon', help='last name')
parser.add_argument('--school', type=str, required=True, default='', help='school name')  # required : Must enter parameters
args = parser.parse_args()

#print(args.integers)
#print(args.integers_many)
#print(sum(args.sum))
print(args.family + args.name + ' ' + 'from' + ' ' + args.school+ ' ' +'university')
print(args)