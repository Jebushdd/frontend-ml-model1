import pickle

print('Hola! Este es el lector de modelos de prediccion de clientes')
modelo = 'C:\\Users\\Jesus\\Documents\\Proyectos\\Maven_Analytics\\Churn-Challenge\\programa\\modelo.bin'
print()
print('Estamos cargando tu modelo')
with open(modelo, 'rb') as cargar_modelo:
    dv, model = pickle.load(cargar_modelo)
print()
print(f'Tu diccionario es {dv}')
print()
print(f'Tu modelo es {model}')
print()
print('Presiona enter para cerrar el script')
input()
