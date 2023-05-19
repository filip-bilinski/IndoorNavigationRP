import pymongo

class db_handler:

    def __init__(self, client, database_name):
        self.data_base = client[database_name]

    def add_entry(self, building, room):
        self.data_base[building].insert_one(room)

    def retrieve_rooms(self):
        result = {}
        for collection in self.data_base.list_collection_names():
            coursor = self.data_base[collection].find({})
            intermediate = []
            for document in coursor:
                intermediate.append(document['room'])
            
            result.update({collection: intermediate})
        
        return result