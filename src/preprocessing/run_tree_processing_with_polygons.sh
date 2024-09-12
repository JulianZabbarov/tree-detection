python src/preprocessing/cluster_trees_with_polygons.py -f data/sauen/treelocations
python src/preprocessing/create_bboxs_using_polygons.py -f data/sauen/treelocations/with_polygons
python src/preprocessing/filter_bboxs.py -t 0.25 -s 10 -f data/sauen/treelocations/with_polygons/treeDetails-20230720_Sauen_PLS_clustered_with_polygon-bboxs_using_polygons.geojson