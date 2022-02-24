#!/usr/bin/env python3
# author: R. Hubert
# email: hubert@cl.uni-heidelberg.de

import xml.etree.cElementTree as ET

tree = ET.parse('../en-de.bicleaner07.tmx')
root = tree.getroot()

print(root.find('de'))
