# %%
import socket
import json

# %%

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 50000))
print("Connection with the server is established")
print("--------------------------------")

while True:
    d_input = dict()
    d_input['price_eth'] = float(input('Input etherium price:'))
    d_input['background'] = input('Input background color:')
    d_input['eyes'] = input('Input eyes type:')
    d_input['mouth'] = input('Input mouth type:')
    d_input['clothes'] = input('Input clothes type:')

    data = json.dumps([d_input]).encode()
    s.send(data)
    print("The data is sent to the server")

    data = s.recv(1024)
    print ('Estimated price', data.decode('utf-8')[1:-1])
    print('Try again?')

s.close()

# %%
