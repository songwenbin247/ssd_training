from opcua import Client
import json

## url = {0:{'url': 'opc.tcp:...', 'open':'yes', 'connect': client}}
## opc_id = {'name':{'opc_id': url_index, 'node_id': 'ns = 2; i = "relay_AR"', 'nodeHander': nodehander}}

class opc_client():
    def __init__(self, path):
        self.path = path
        self.load_opc_info()
        self.open_connect_and_get_node_all()

    opc_info = {}
    url = {}
    opc_id = {}

    def load_opc_info(self):
        with open(self.path , 'r') as fd:
            info = fd.read()
            self.opc_info = json.loads(info)

    def open_connect_and_get_node_all(self):
        i = 0
        for name, node in self.opc_info.items():

            try:
                index = [x.values()[0] for x in self.url.values()].index(node['url'])

            except ValueError:
                self.url.update({i:{'url':node['url']}})
                index = i
                i = i + 1
            self.opc_id.update({name: {"opc_id": index, "node_id": node['node']}})

        for id_key, url in self.url.items():
            client = Client(url['url'])
            try:
                client.connect()
                url['open'] = 'yes'
                url['connect'] = client
            except:
                url['open'] = 'no'
        for name, node in self.opc_id.items():
            try:
                client = self.url[node['opc_id']]['connect']
                nodeHander = client.get_node(node['node_id'].encode('ascii'))
                node["nodeHander"] = nodeHander
            except:
                node["nodeHander"] = None

    def get_value(self, node_name):

        try:
            value = self.opc_id[node_name]['nodeHander'].get_value()
        except:
            value = None

        return value
    def opc_destroy(self):
        for id, url in self.url.items():
            try:
                url['connect'].disconnect()
            except:
                pass
    def __del__(self):
        for id, url in self.url.items():
            try:
                url['connect'].disconnect()
            except:
                pass

if __name__ == "__main__":
    A = opc_client("/home/songwb/samba/ar/photo/ssd_training/opc_ua.config")

    for name in A.opc_info.keys():
        str = A.get_value(name)
        print("%s is %s" % (name, str))
