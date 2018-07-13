from pyfacy import face_clust


mdl = face_clust.Face_Clust_Algorithm('./cluster')
mdl.load_faces()
mdl.save_faces('./cluster_output')
