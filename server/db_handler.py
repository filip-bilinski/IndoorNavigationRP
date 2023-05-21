import pymongo

class db_handler:

    def __init__(self, client, database_name):
        self.data_base = client[database_name]

    def add_entry(self, building, room):
        self.data_base[building].insert_one(room)

    def retrieve_rooms(self):
        result = {}
        number_of_rooms = 0
        for collection in self.data_base.list_collection_names():
            coursor = self.data_base[collection].find({})
            intermediate = []
            for document in coursor:
                if document['room'] not in intermediate:
                    number_of_rooms += 1
                    intermediate.append(document['room'])
            
            result.update({collection: intermediate})
        
        return result, number_of_rooms