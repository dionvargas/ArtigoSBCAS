import os.path

class DataManipulator:

    def __init__(self, path_data):
        self.path_data = path_data

    def create_data(self):
        if os.path.exists(self.path_data):
            os.remove(self.path_data)
        open(self.path_data, "w")

    def remove_data(self):
        if os.path.exists(self.path_data):
            os.remove(self.path_data)

    def save_data(self, dict, folder, titles=True):
        if not os.path.exists(self.path_data):
            file = open(self.path_data, "w")
            if titles:
                first = True
                for data in dict:
                    if first:
                        file.write("Class;")
                    else:
                        file.write(";")
                    file.write(str(data))
                    first = False
                file.write("\n")
            file.close()

        file = open(self.path_data, "a")
        first = True
        for data in dict:
            if first:
                file.write(str(folder) + ';')
            else:
                file.write(";")
            file.write(str(dict[data]))
            first = False
        file.write("\n")
        file.close()

    def read_data(self, h_titles=True, type=str):
        file = open(self.path_data, 'r')

        titles = []
        data = []

        if h_titles:
            titles = file.readline()[:-1].split(";")

        for line in file:
            data.append(list(map(type, line[:-1].split(";"))))

        file.close()

        return titles, data
