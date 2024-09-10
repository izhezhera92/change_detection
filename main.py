__author__ = "Ivan Zhezhera"
__date__ = "09.09.2024"


import os
import cv2
import glob
import torch
import rasterio
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from affine import Affine
import matplotlib.pyplot as plt

import datetime

import rasterio
from rasterio.enums import Resampling
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin, from_gcps

from scipy.spatial import Delaunay, delaunay_plot_2d


from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd


import calculate_utils as cu



parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--image1', default="./odm_orthophoto_small.tif", required=False, help="")
parser.add_argument('-i2', '--image2', default="./odm_orthophoto_big.tif", required=False, help="")
parsed = parser.parse_args()

image1_path = parsed.image1
image2_path = parsed.image2



class ortho_transformation(object):
    def __init__(self, auto_clean = False):

        self.device_type  = 'cpu' # 'mps'
        if auto_clean:
            self.__clean_old_files()
        
        self.min_x = 100000
        self.max_x = 0
        self.min_y = 100000
        self.max_y = 0

        self.p_min_x = []
        self.p_max_x = []
        self.p_min_y = []
        self.p_max_y = []

        self.p_min_x_current = []
        self.p_max_x_current = []
        self.p_min_y_current = []
        self.p_max_y_current = []

        self.src_crs = {'init': 'epsg:32636'}

        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)
        self.max_num_keypoints = 2048
        self.verbose = True

    def pix2coord(self, path = None, verbose = True):
        ''' Method for geo data importing from geotif'''
        if path is not None:
            with rasterio.open(path) as src:
                band1 = src.read(1)
                if self.verbose:
                    print('[info] Band1 has shape', band1.shape)
                height = band1.shape[0]
                width = band1.shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                lons = np.array(xs)
                lats = np.array(ys)
            return lons, lats
        return 0, 0

    def __clean_old_files(self, path = './pairs/*'):
        ''' Method for cleaning old files '''
        files = glob.glob(path)
        for f in files:
            os.remove(f)
        if self.verbose:
            print("[info] Cleaning is done")

    def __write_geotif_image(self, path = None, data = None, meta = None):
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data)
            if self.verbose:
                print("[info] tif file: {path} saved")
        return 0

    def __image_resizing(self, path = "./ortho_second.jpg", fx = 0.1, fy = 0.1, quality_persent = 100):
        ''' Method for the image resizing '''
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_smaller = cv2.resize(img, (0,0), fx=fx,fy=fy)
        cv2.imwrite("new_ortho_second.jpg", img_smaller, [cv2.IMWRITE_JPEG_QUALITY, quality_persent])
        return 0


    def matchec_detection(self, image0, image1):
        ''' Method for the matches finding with LightGlue algorithm'''

        extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)  
        matcher = LightGlue(features="superpoint").eval().to(self.device)

        feats0 = extractor.extract(image0.to(self.device))
        feats1 = extractor.extract(image1.to(self.device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return m_kpts0, m_kpts1, matches01, kpts0, kpts1


    def affine_transformation_simple(self, img_for_tranformation = None, img_full = None, pts1 = None, pts2 = None, edge = 20):
        #print("image: ", type(img))
        rows, cols, ch = img_full.shape
        if (rows >=edge and cols >= edge):
            #print("shape: ", rows, cols, ch)
            print("pts1, pts2: ", pts1, pts2)

            #cv2.imwrite("./pairs/part.jpg", img_for_tranformation, [cv2.IMWRITE_JPEG_QUALITY, 100])

            M = cv2.getAffineTransform(pts2, pts1)
            dst = cv2.warpAffine(img_for_tranformation, M, (cols, rows))
            return dst

        else:
            return None

    def affine_transformation_gcp(self, path = None, gcps = None):
        ''' Method for geotif transformation via GCP's'''
        with rasterio.open(path) as src:
            transformer = rasterio.transform.GCPTransformer(gcps=gcps) 
            transform = from_gcps(gcps)

            image_data = src.read(
                out_shape=(
                    src.count,
                    int(src.height),
                    int(src.width)
                ),
                resampling=Resampling.nearest
            )

            new_meta = src.meta.copy()
            new_meta.update({
                'crs': self.src_crs,      
                'transform': transform     
            })

            preffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
            res = self.__write_geotif_image(path = str('./') + str(preffix) + '.tif', data = image_data, meta = new_meta)
            return 0

        return 1

    def setup_faraway_gcps_xy(self, x_big, y_big, x_small, y_small):
        ''' Method for the control points searching'''
        if (x_big<self.min_x):
            self.min_x = x_big
            self.p_min_x = [x_big, y_big]
            self.p_min_x_current = [x_small, y_small]

        elif (x_big>self.max_x):
            self.max_x = x_big
            self.p_max_x = [x_big, y_big]
            self.p_max_x_current = [x_small, y_small]

        if(y_big<self.min_y):
            self.min_y = y_big
            self.p_min_y = [x_big, y_big]
            self.p_min_y_current = [x_small, y_small]

        elif(y_big>self.max_y):
            self.max_y = y_big
            self.p_max_y = [x_big, y_big]
            self.p_max_y_current = [x_small, y_small]


    def get_wgs84_faraway_gcps(self, lats_big, lon_big, p1_xy, p2_xy, p3_xy, p4_xy):

        p1_wgs84 = [lons_big[p1_xy[0]][p1_xy[1]], lats_big[p1_xy[0]][p1_xy[1]]]
        p2_wgs84 = [lons_big[p2_xy[0]][p2_xy[1]], lats_big[p2_xy[0]][p2_xy[1]]]
        p3_wgs84 = [lons_big[p3_xy[0]][p3_xy[1]], lats_big[p3_xy[0]][p3_xy[1]]]
        p4_wgs84 = [lons_big[p4_xy[0]][p4_xy[1]], lats_big[p4_xy[0]][p4_xy[1]]]        
        return p1_wgs84, p2_wgs84, p3_wgs84, p4_wgs84

    def get_ploygon_frome_fullframe(self, img = None, pts = None):
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img                         #<<<<<<<<< comment for small
        #croped = img[y:y+h, x:x+w].copy()   <<<<<<<<< uncomment for small

        ## (2) make mask
        #pts = pts - pts.min(axis=0)         <<<<<<<<< uncomment for small

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        return mask, dst


    def vizual(self, image0, image1, m_kpts0, m_kpts1, matches01):
        axes = viz2d.plot_images([image0, image1])

        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.7)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
        viz2d.save_plot("./res1.jpg")

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

        viz2d.save_plot("./res2.jpg")
        return 0



if __name__ == "__main__":
    ot = ortho_transformation()
    
    lons_small, lats_small = ot.pix2coord(path = image1_path)
    print("[info] small ortho had read")

    lons_big, lats_big = ot.pix2coord(path = image2_path)
    print("[info] big ortho had read")


    image0 = load_image(image2_path)
    image1 = load_image(image1_path)


    m_kpts0, m_kpts1, matches01, kpts0, kpts1 = ot.matchec_detection(image0 = image0, image1 = image1)

    #vizual(image0, image1, m_kpts0, m_kpts1, matches01)
    
    wear_angles = []
    distances = []
    


    for i in tqdm(range(len(m_kpts0))):
        lat_big = round(lats_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)] / 100000 ,6)
        lon_big = round(lons_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)] / 10000 ,6)
        lat_small = round(lats_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 100000,6)
        lon_small = round(lons_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 10000,6)

        if lat_big < 10:
            lat_big = lat_big + 50.0

        ot.setup_faraway_gcps_xy(x_big = m_kpts0[i][1].numpy().astype(int), y_big = m_kpts0[i][0].numpy().astype(int), x_small = m_kpts1[i][1].numpy().astype(int), y_small = m_kpts1[i][0].numpy().astype(int)) 



        #print(lat_big, lon_big, lat_small, lon_small, m_kpts0[i][1].numpy().astype(int), m_kpts0[i][0].numpy().astype(int), m_kpts1[i][1].numpy().astype(int), m_kpts1[i][0].numpy().astype(int))
        wear_angle = cu.Geodesy().calculate_wear_angle(lat1 = lat_big, lon1 = lon_big, lat2 = lat_small, lon2 = lon_small)
        distance = cu.Geodesy().calculate_distance_wgs84(lat1 = lat_big, lon1 = lon_big, lat2 = lat_small, lon2 = lon_small)
        #print(wear_angle)
        wear_angles.append(wear_angle)
        #distances.append(distance)


    p1_wgs84, p2_wgs84, p3_wgs84, p4_wgs84 = ot.get_wgs84_faraway_gcps(lats_big, lons_big, ot.p_min_x, ot.p_max_x, ot.p_min_y, ot.p_max_y)


    gcps = [
    GroundControlPoint(ot.p_min_x_current[0], ot.p_min_x_current[1], p1_wgs84[0], p1_wgs84[1]),
    GroundControlPoint(ot.p_max_x_current[0], ot.p_max_x_current[1], p2_wgs84[0], p2_wgs84[1]),
    GroundControlPoint(ot.p_min_y_current[0], ot.p_min_y_current[1], p3_wgs84[0], p3_wgs84[1]),
    GroundControlPoint(ot.p_max_y_current[0], ot.p_max_y_current[1], p4_wgs84[0], p4_wgs84[1])
]

    ot.affine_transformation_gcp(gcps = gcps, path = image1_path)

    wears_np = np.array(wear_angles) 
    distances_np = np.array(distances) 
    #print(wears_np)
    _, bins = pd.cut(wears_np, bins=200, retbins=True)
    plt.hist(wears_np, bins)

    
    '''
    # https://docs.scipy.org/doc//scipy-1.12.0/reference/generated/scipy.spatial.Delaunay.html
    tri = Delaunay(m_kpts0)
    _ = delaunay_plot_2d(tri)



    img_1_cv2 = cv2.imread(image2_path)
    img_2_cv2 = cv2.imread(image1_path)

    for i in tqdm(range(len(m_kpts0[tri.simplices]))):

        pts1 = (np.rint(m_kpts0[tri.simplices][i].cpu().detach().numpy())).astype(int)
        pts2 = (np.rint(m_kpts1[tri.simplices][i].cpu().detach().numpy())).astype(int)

        mask_1, dst_1 = ot.get_ploygon_frome_fullframe(img = img_1_cv2, pts = pts1)
        mask_2, dst_2 = ot.get_ploygon_frome_fullframe(img = img_2_cv2, pts = pts2)
        #cv2.imwrite("./mask.png", mask)
        #cv2.imwrite("./dst.png", dst)

        #transformed_image = affine_transformation_simple(img_for_tranformation = dst_2, img_full = dst_1, pts1 = pts1, pts2 = pts2)

        #if transformed_image is not None:

        #    cv2.imwrite("./pairs/" + str(i) + "part_src.jpg", img_1_cv2, [cv2.IMWRITE_JPEG_QUALITY, 100])
        #    cv2.imwrite("./pairs/" + str(i) + "part_transformered.jpg", transformed_image, [cv2.IMWRITE_JPEG_QUALITY, 100]) 


        cv2.imwrite("./pairs/" + str(i) + "part_src.jpg", dst_1, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite("./pairs/" + str(i) + "part_transformered.jpg", dst_2, [cv2.IMWRITE_JPEG_QUALITY, 100])'''
        
    plt.show()
    





