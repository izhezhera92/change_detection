__author__ = "Ivan Zhezhera"
__date__ = "09.09.2024"


import os
import cv2
import glob
import torch
import logging
import datetime
import rasterio
import argparse
import numpy as np
from pandas import cut 
from tqdm import tqdm
from affine import Affine
import matplotlib.pyplot as plt




import rasterio
from rasterio.enums import Resampling
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin, from_gcps

from scipy.spatial import Delaunay, delaunay_plot_2d


from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd


import calculate_utils as cu



parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--image1', default="./src/odm_orthophoto_small.tif", required=False, help="")
parser.add_argument('-i2', '--image2', default="./src/odm_orthophoto_big.tif", required=False, help="")
parsed = parser.parse_args()

image1_path = parsed.image1
image2_path = parsed.image2



class ortho_transformation(object):
    def __init__(self, auto_clean = False):

        self.device_type  = 'cpu' # 'mps'
        if auto_clean:
            self.__clean_old_files()

        self.src_crs = {'init': 'epsg:32636'}

        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)
        self.max_num_keypoints = 2048
        self.verbose = True

        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')  


    def pix2coord(self, path: str = '', verbose: bool = True)-> list:
        ''' Method for geo data importing from geotif'''
        if path is not None:
            with rasterio.open(path) as src:
                band1 = src.read(1)
                if self.verbose:
                    logging.info(f"{path} has shape: {band1.shape}")
                height = band1.shape[0]
                width = band1.shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                lons = np.array(xs)
                lats = np.array(ys)
            return lons, lats
        return [], []


    def __clean_old_files(self, path: str = './pairs/*')-> int:
        ''' Method for cleaning old files '''
        files = glob.glob(path)
        for f in files:
            os.remove(f)
        if self.verbose:
            logging.info(f"Cleaning is done")

        return 0

    def __write_geotif_image(self, path: str = None, data = None, meta = None)-> int:
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data)
            if self.verbose:
                logging.info(f"tif file: {path} have been saved")
        return 0

    def __image_resizing(self, path = "./ortho_second.jpg", fx = 0.1, fy = 0.1, quality_persent = 100)-> int:
        ''' Method for the image resizing '''
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_smaller = cv2.resize(img, (0,0), fx=fx,fy=fy)
        cv2.imwrite("new_ortho_second.jpg", img_smaller, [cv2.IMWRITE_JPEG_QUALITY, quality_persent])
        if self.verbose:
            logging.info(f"image: {path} have been resized")
        return 0

    def matchec_detection(self, image0, image1)-> tuple:
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
        if self.verbose:
            logging.info(f"Detected: {len(matches)} pairs")
        return m_kpts0, m_kpts1, matches01, kpts0, kpts1

    def affine_transformation_simple(self, img_for_tranformation = None, img_full = None, pts1 = None, pts2 = None, edge = 20):
        
        rows, cols, ch = img_full.shape
        if (rows >= edge and cols >= edge):
            M = cv2.getAffineTransform(pts2, pts1)
            dst = cv2.warpAffine(img_for_tranformation, M, (cols, rows))
            return dst

        else:
            return None

    def __image_transformatio(self, src: rasterio.io.DatasetReader = None):
        image_data = src.read(
                out_shape=(
                    src.count,
                    int(src.height),
                    int(src.width)
                ),
                resampling=Resampling.nearest
            )
        return image_data

    def __update_meta(self, src: str = None, transform: Affine = None):
        new_meta = src.meta.copy()
        new_meta.update({
                'crs': self.src_crs,      
                'transform': transform     
            })
        return new_meta

    def gcp_transformation(self, path: str = None, gcps: list = None, image_gen: bool = True)-> int: # | np.ndarray 
        ''' Method for geotif transformation via GCP's'''
        with rasterio.open(path) as src:
            transformer = rasterio.transform.GCPTransformer(gcps=gcps) 
            transform = from_gcps(gcps)

            image_data = self.__image_transformatio(src = src)
            logging.info(f"Image has been transformated")
            new_meta = self.__update_meta(src = src, transform = transform)
            preffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
            if image_gen:
                res = self.__write_geotif_image(path = str('./res/') + str(preffix) + '.tif', data = image_data, meta = new_meta)
                if self.verbose:
                    logging.info(f"Image has been saved to: {str('./res/') + str(preffix) + '.tif'}")

            else:
                if self.verbose:
                    logging.info(f"Image has transformated")
                return image_data
            
            return 0
        if self.verbose:
            logging.warning(f"Image transformation has not done!")
        return 1

        

    def get_ploygon_frome_fullframe(self, img: np.ndarray = None, pts: list = None):
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


    def __vizual(self, image0: np.ndarray, image1: np.ndarray, m_kpts0: list, m_kpts1: list, matches01: list, kpts0: list, kpts1: list):
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.7)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
        viz2d.save_plot("./res1.jpg")
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
        viz2d.save_plot("./res2.jpg")
        return 0




    def scenario_process(self):
        try:
            lons_small, lats_small = self.pix2coord(path = image1_path)
            logging.info(f"small ortho had been read")

            lons_big, lats_big = self.pix2coord(path = image2_path)
            logging.info(f"big ortho had been read")


            image0 = load_image(image2_path)
            image1 = load_image(image1_path)

            (m_kpts0, m_kpts1, matches01, kpts0, kpts1) = self.matchec_detection(image0 = image0, image1 = image1)

            self.__vizual(image0, image1, m_kpts0, m_kpts1, matches01, kpts0, kpts1)
            
            wear_angles = []
            distances = []
            gcps = []


            for i in tqdm(range(len(m_kpts0))):
                lat_big = round(lats_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)] / 100000 ,6)
                lon_big = round(lons_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)] / 10000 ,6)
                lat_small = round(lats_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 100000,6)
                lon_small = round(lons_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 10000,6)

                if lat_big < 10:
                    lat_big = lat_big + 50.0

                wear_angle = cu.Geodesy().calculate_wear_angle(lat1 = lat_big, lon1 = lon_big, lat2 = lat_small, lon2 = lon_small)
                distance = cu.Geodesy().calculate_distance_wgs84(lat1 = lat_big, lon1 = lon_big, lat2 = lat_small, lon2 = lon_small)
                
                wear_angles.append(wear_angle)
                distances.append(distance)
                
                p1 = m_kpts1[i][1].numpy().astype(int)
                p2 = m_kpts1[i][0].numpy().astype(int)
                p3 = lons_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)]
                p4 = lats_big[m_kpts0[i][1].numpy().astype(int)][m_kpts0[i][0].numpy().astype(int)]
                gcps.append(GroundControlPoint(p1, p2, p3, p4))


            self.gcp_transformation(gcps = gcps, path = image1_path)

            wears_np = np.array(wear_angles) 
            distances_np = np.array(distances) 
            _, bins = cut(wears_np, bins=250, retbins=True)
            _, bins = cut(distances_np, bins=250, retbins=True)
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

        except Exception as ex:
            if self.verbose:
                logging.critical(f"Transformation failed!")
                logging.critical(f"{ex}")
            
            pass






if __name__ == "__main__":
    ot = ortho_transformation()
    ot.scenario_process()
    
    
    





