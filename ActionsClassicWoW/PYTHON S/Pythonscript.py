import os
import json
import mysql.connector
import logging

# Configuración de logs
logging.basicConfig(filename='migration_log.txt', level=logging.INFO)

def migrate_data(json_directory, table_name, conn):
    cursor = conn.cursor()

    # Log de inicio
    logging.info(f"Iniciando migración de datos para la tabla {table_name}...")

    try:
        # Iterar sobre todos los archivos JSON en el directorio
        for filename in os.listdir(json_directory):
            if filename.endswith(".json"):
                logging.info(f"Procesando archivo: {filename}")
                with open(os.path.join(json_directory, filename), 'r') as file:
                    try:
                        # Cargar datos JSON desde el archivo
                        data = json.load(file)
                        print(f"Datos cargados desde {filename}: {data}")

                        # Acceder a los datos de la subasta anidados bajo la clave 'item'
                        item_data = data.get('item', {})  # Si 'item' no está presente, se establece un diccionario vacío

                        # Usar el ID único del archivo como identificador de subasta
                        auction_id = data.get('id')

                        if auction_id is not None:
                            # Verificar si la entrada ya existe en la base de datos
                            query_check_duplicate = f"SELECT 1 FROM {table_name} WHERE auction_id = %s"
                            cursor.execute(query_check_duplicate, (auction_id,))
                            result = cursor.fetchone()

                            if result:
                                # Entrada duplicada, puedes considerar la actualización.
                                logging.warning(f"Entrada duplicada encontrada para {filename}. Puedes considerar la actualización.")
                            else:
                                # Construir consulta de inserción con parámetros
                                query_insert = f"""
                                INSERT INTO {table_name} (auction_id, item_id, buyout, bid, quantity, time_left)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """

                                # Ejecutar consulta de inserción con parámetros
                                cursor.execute(query_insert, (
                                    auction_id,
                                    item_data.get('id', None),
                                    data.get('buyout', None),
                                    data.get('bid', None),
                                    data.get('quantity', None),
                                    data.get('time_left', None)
                                ))

                                # Confirmar cambios después de procesar el archivo (opcional)
                                conn.commit()

                                print(f"Inserción exitosa para {filename}")
                        else:
                            logging.warning(f"Clave 'id' no encontrada o es None en el archivo {filename}")

                    except (json.JSONDecodeError, mysql.connector.Error, KeyError) as e:
                        logging.error(f"Error en el archivo {filename}: {e}")

        # Confirmar cambios al final del procesamiento
        conn.commit()

        # Log de finalización
        logging.info(f"Migración de datos para la tabla {table_name} completada.")

    except mysql.connector.Error as err:
        # Manejar errores de conexión o ejecución de consultas SQL
        logging.error(f"Error de MySQL: {err}")

    finally:
        # Cerrar conexión y cursor en el bloque 'finally' para garantizar que se cierre incluso si ocurre una excepción
        cursor.close()
        conn.close()

# Conexión a la base de datos MySQL
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="12792",  # Tu contraseña aquí
    database="AuctionDB"  # Cambiado al nombre de la base de datos correcto
)

# Ruta al directorio que contiene los archivos JSON
json_directory = "/home/cmiranda/Documents/ActionsClassicWoW/WOW Sql/05-11-2023"

# Migrar datos para la tabla Auctions
migrate_data(json_directory, "Auctions", conn)
