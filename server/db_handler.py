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
            if collection == "room_to_label":
                continue
            coursor = self.data_base[collection].find({})
            intermediate = []
            for document in coursor:
                if document['room'] not in intermediate:
                    number_of_rooms += 1
                    intermediate.append(document['room'])
            
            result.update({collection: intermediate})
        
        return result, number_of_rooms

    def prepare_training_dataset(self):
        labels = []
        data = []
        room_to_label = {}

        counter = 0

        for collection in self.data_base.list_collection_names():
            if collection == "room_to_label":
                continue
            coursor = self.data_base[collection].find({})
            for document in coursor:
                string_label = collection + "_" + document['room']
                if string_label not in room_to_label:
                    room_to_label.update({string_label: counter})
                    
                    labels.append(counter)
                    data.append(document['audio'])

                    counter += 1
                else:
                    labels.append(room_to_label[string_label])
                    data.append(document['audio'])

        self.data_base['room_to_label'].delete_many({})
        self.data_base['room_to_label'].insert_one(room_to_label)

        return labels, data