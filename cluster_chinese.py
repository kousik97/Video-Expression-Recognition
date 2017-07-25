from sklearn import cluster
import cv2
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
#from matplotlib import pyplot as plt
import argparse
import shutil
predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
#faces_folder_path = "/home2/rajib/face/test9"
result_dir="/root/video_emotion/Results/output"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
#print (detector)
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
def face_distance(face_encodings, face_to_compare):
    
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def _chinese_whispers(encoding_list, threshold, iterations):
    """ 

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    # from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx + 1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx + 1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx + 1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': (1-distance)}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):

                    if G.node[ne]['cluster'] in clusters:

                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0

            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters


def cluster_facial_encodings(facial_encodings):
    """ 
        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest
    """

    if len(facial_encodings) <= 1:
        print("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items(),threshold=0.50,iterations=15)
    return sorted_clusters


#win = dlib.image_window()
def face_feature(frame): 
    #print("Processing file: {}".format(frame))
    img = io.imread(frame)

    #win.clear_overlay()
    #win.set_image(img)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.

    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        #win.clear_overlay()
        #win.add_overlay(d)
        #win.add_overlay(shape)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        a=np.array(face_descriptor)
        #print(a)
        facial_encodings[frame]=a
   

        #dlib.hit_enter_to_continue()

def save(facial_encodings):
	print(len(facial_encodings))
	answer =cluster_facial_encodings(facial_encodings)
        try:
            shutil.rmtree(result_dir, ignore_errors=True)
            os.mkdir(result_dir) 
        except:
            os.mkdir(result_dir)
	for i in range(len(answer)):
	    path = result_dir+"/"+"pic"+ str(i + 1)
	    os.mkdir(path)
	for k,d in enumerate(answer):
	    #j = 0
	    for elem in d:
		img=io.imread(elem)
                pic_name=elem.split("/")[-1]
                save_path=result_dir+"/pic{}/"+pic_name
		io.imsave(save_path.format(k+1),img)
		#j += 1
        print("Completed")

def traverse_dir(dst_path):
	# generate features
	for root, subdirs, files in os.walk(dst_path):
		for f in files:
			img = os.path.join(root, f)
			face_feature(img)
			

def parse_args(args):
    if args["image"] and os.path.isfile(args["image"]):
        img = args["image"]
        face_feature(img)

    elif args["dir"] and os.path.isdir(args["dir"]):
        dst_path = args["dir"]
        traverse_dir(dst_path)
    else:
        print("path not found")

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    facial_encodings={}
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input directory of images")
    ap.add_argument("-d", "--dir", help="path to the dir of image files")
    ap.add_argument("-v", "--vid", help="video source", action='store_true')

    args = vars(ap.parse_args())

    if args["image"] or args["dir"]:
        parse_args(args)
        save(facial_encodings)
    else:
        print("python classify.py [-d <photo dir> | -i <photo file>]")
    
