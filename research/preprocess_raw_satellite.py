
import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling

BANDS=('B03','B04','B08','B11')

def load_tiff(z,band,date):
    names=[n for n in z.namelist() if f'_{band}_(Raw).tiff' in n and date in n]
    if len(names)!=1:
        raise ValueError(f'Expected exactly one {date} {band} raw TIFF; found {names}')
    with MemoryFile(z.read(names[0])) as mf:
        with mf.open() as ds:
            return ds.read(1,masked=True).filled(np.nan).astype('float32'),ds.transform,ds.crs

def safe_index(green,other,mask):
    den=green+other
    out=np.full(green.shape,np.nan,dtype='float32')
    np.divide(green-other,den,out=out,where=mask & np.isfinite(den) & (np.abs(den)>1e-6))
    return out

def run(before_zip,after_zip,output):
    with ZipFile(before_zip) as z0, ZipFile(after_zip) as z1:
        ref,ref_transform,ref_crs=load_tiff(z0,'B03','2026-08-12')
        before={};after={}
        for band in BANDS:
            x,transform,crs=load_tiff(z0,band,'2026-08-12')
            if x.shape!=ref.shape or transform!=ref_transform or crs!=ref_crs:
                aligned=np.full(ref.shape,np.nan,dtype='float32')
                reproject(x,aligned,src_transform=transform,src_crs=crs,src_nodata=np.nan,
                          dst_transform=ref_transform,dst_crs=ref_crs,dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
                x=aligned
            before[band]=x
            y,y_transform,y_crs=load_tiff(z1,band,'2026-10-13')
            aligned=np.full(ref.shape,np.nan,dtype='float32')
            reproject(y,aligned,src_transform=y_transform,src_crs=y_crs,src_nodata=np.nan,
                      dst_transform=ref_transform,dst_crs=ref_crs,dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            after[band]=aligned
    valid=np.ones(ref.shape,dtype=bool)
    for band in BANDS:valid &= np.isfinite(before[band]) & np.isfinite(after[band])
    ndwi_before=safe_index(before['B03'],before['B08'],valid)
    ndwi_after=safe_index(after['B03'],after['B08'],valid)
    ndsi_before=safe_index(before['B03'],before['B11'],valid)
    ndsi_after=safe_index(after['B03'],after['B11'],valid)
    payload={**{f'before_{b}':before[b] for b in BANDS},
             **{f'after_{b}':after[b] for b in BANDS},
             'ndwi_before':ndwi_before,'ndwi_after':ndwi_after,
             'ndsi_before':ndsi_before,'ndsi_after':ndsi_after,
             'ndwi_change':ndwi_after-ndwi_before,'ndsi_change':ndsi_after-ndsi_before,
             'valid_mask':valid}
    np.savez_compressed(output,**payload)
    print('Saved:',output,'valid pixels:',valid.sum(),'coverage %:',round(100*valid.mean(),2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--before',required=True,type=Path)
    p.add_argument('--after',required=True,type=Path)
    p.add_argument('--output',default='glacierguard_data.npz',type=Path)
    a=p.parse_args()
    run(a.before,a.after,a.output)
