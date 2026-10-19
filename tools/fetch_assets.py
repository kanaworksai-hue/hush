"""Download the CC0 assets the demo scenes use into public/assets/ (not committed).

    py tools/fetch_assets.py

Sources (all CC0):
  - Poly Haven models + textures + HDRIs (https://polyhaven.com, CC0)
  - RobotExpressive.glb by Tomas Laulhe (three.js examples, CC0)
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public', 'assets')
UA = {'User-Agent': 'hush-fetch/0.1'}
RES = '1k'

MODELS = [
    # living room
    'Sofa_01', 'ArmChair_01', 'CoffeeTable_01', 'potted_plant_04', 'wooden_bookshelf_worn',
    'book_encyclopedia_set_01', 'ceramic_vase_01', 'side_table_01', 'Television_01', 'throw_pillows_01',
    'tea_set_01', 'wine_bottles_01', 'modern_ceiling_lamp_01', 'Chandelier_01',
    # courtyard
    'wooden_picnic_table', 'Barrel_01', 'wooden_crate_01', 'garden_gnome', 'boulder_01', 'tree_stump_01',
    'potted_plant_02', 'street_lamp_01', 'concrete_cat_statue', 'fir_sapling', 'shrub_01', 'planter_box_01',
    # box
    'marble_bust_01', 'brass_vase_01', 'rubber_duck_toy',
]
TEXTURES = ['wood_floor_worn', 'plastered_wall_02', 'cobblestone_floor_01', 'grey_plaster', 'brick_wall_001',
            'beige_wall_001', 'herringbone_parquet']
HDRIS = ['hilly_terrain_01', 'spaichingen_hill']  # courtyard (golden hour), living room (afternoon)
HDRI_RES = '2k'
EXTRA = {
    'RobotExpressive/RobotExpressive.glb':
        'https://raw.githubusercontent.com/mrdoob/three.js/r170/examples/models/gltf/RobotExpressive/RobotExpressive.glb',
}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def save(rel, url):
    path = os.path.join(ROOT, rel)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = get(url)
    with open(path, 'wb') as f:
        f.write(data)
    print('  ', rel, len(data) // 1024, 'KB')


def main(MODELS=MODELS, TEXTURES=TEXTURES, HDRIS=HDRIS, EXTRA=EXTRA):
    for m in MODELS:
        files = json.loads(get(f'https://api.polyhaven.com/files/{m}'))
        g = files['gltf'][RES]['gltf']
        print(m)
        save(f'polyhaven/{m}/{m}.gltf', g['url'])
        for rel, inc in g['include'].items():
            save(f'polyhaven/{m}/{rel}', inc['url'])
    for t in TEXTURES:
        files = json.loads(get(f'https://api.polyhaven.com/files/{t}'))
        print(t)
        for key, short in (('Diffuse', 'diff'), ('nor_gl', 'nor'), ('arm', 'arm')):
            if key in files:
                save(f'textures/{t}/{t}_{short}.jpg', files[key][RES]['jpg']['url'])
    for h in HDRIS:
        files = json.loads(get(f'https://api.polyhaven.com/files/{h}'))
        print(h)
        save(f'hdri/{h}_{HDRI_RES}.hdr', files['hdri'][HDRI_RES]['hdr']['url'])
    for rel, url in EXTRA.items():
        print(rel)
        save(rel, url)
    print('done')


if __name__ == '__main__':
    sys.exit(main())
