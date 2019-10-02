from __future__ import print_function
import numpy as np
import scipy.ndimage
import pylab as p
import glob
import os

def plot1dslice(prefix="output/",ps="-",slice_z=None,slice_y=None,maxgrid=2,vmin=-0.15,vmax=0.15,thisgrid=0,
                zoom_pad_cells=3,diff_prefix=None,offset=None):
    pad_cells = zoom_pad_cells if thisgrid>0 else 0

    a = _load_grid(prefix, diff_prefix, thisgrid)

    ax,ay,az,aL = [float(x) for x in open(prefix+"grid-info-%d.txt"%thisgrid).readline().split()]

    if offset is None:
        offset = aL/2

    if slice_z is None:
        slice_z = aL/2 + (aL/len(a))/2

    if slice_y is None:
        slice_y = aL/2 + (aL/len(a))/2

    a_sl_z = int(len(a)*((slice_z-az)/aL))
    a_sl_y = int(len(a)*((slice_y-ay)/aL))

    if a_sl_y<0 or a_sl_y>=len(a) or a_sl_z<0 or a_sl_z>=len(a):
        print("Out of range for zoom grid",thisgrid)
        return 0, 0

    dx = aL/len(a)

    a_vals = np.linspace(ax+dx/2+dx*pad_cells,ax+aL-dx/2-dx*pad_cells,len(a)-pad_cells*2)

    if pad_cells>0:
        x_sl = slice(pad_cells,-pad_cells)
    else:
        x_sl = slice(None)

    plot_y_vals = a[x_sl,a_sl_y,a_sl_z].real

    kwargs = {}

    if thisgrid<maxgrid and os.path.exists(prefix+"grid-%d.npy"%(thisgrid+1)):
        xmin, xmax = plot1dslice(prefix,ps,slice_z,slice_y,maxgrid,vmin,vmax,thisgrid+1,zoom_pad_cells,diff_prefix,offset)
        interior_mask = (a_vals>xmin) & (a_vals<xmax)
        l = p.plot(a_vals[interior_mask]-offset,plot_y_vals[interior_mask],":",zorder=-10)
        kwargs['color'] = l[0].get_color()
        plot_y_vals[np.where(interior_mask)[0][1:-1]] = np.nan

    p.plot(a_vals-offset,plot_y_vals,ps,**kwargs)
    p.xlim(0-offset,aL-offset)
    if thisgrid>0:
        return a_vals.min(), a_vals.max()


def plotslice_onegrid_with_wrapping(*args, **kwargs):
    wrap = float(open(args[0]+"grid-info-0.txt").readline().split()[3])
    for x in [-wrap,0,wrap]:
        for y in [-wrap,0,wrap]:
            kwargs['plot_offset']=(x,y)
            X = plotslice_onegrid(*args,**kwargs)
    p.xlim(-wrap/2,wrap/2)
    p.ylim(-wrap/2,wrap/2)
    return X

def _check_and_return_integers(*vals):
    rounded = np.round(vals)
    np.testing.assert_allclose(rounded, vals, atol=1e-3)
    return np.array(rounded,dtype=int)

def _load_grid(prefix,diff_prefix, grid):
    a = np.load(prefix+"grid-%d.npy"%grid)
    ax,ay,az,aL = [float(x) for x in open(prefix+"grid-info-%d.txt"%grid).readline().split()]

    if diff_prefix is not None:
        # load best matching grid
        for diff_grid in range(10,-1,-1): # prefer the highest resolution grid that can be made to match
            bx = None
            try:
                bx,by,bz,bL = [float(x) for x in open(diff_prefix+"grid-info-%d.txt"%diff_grid).readline().split()]
            except IOError:
                continue

            if bL>=aL:
                print("Attempt to compare grid %d in %r to grid %d in %r"%(grid,prefix,diff_grid,diff_prefix))
                break
        if bx is None:
            raise RuntimeError("On grid %d of %r, cannot find a suitable match in %r"%(grid,prefix,diff_prefix))

        b = np.load(diff_prefix+"grid-%d.npy"%diff_grid)
        b_cellsize = bL/len(b)

        # Trim to match
        offset_x = (ax-bx)/b_cellsize
        offset_y = (ay-by)/b_cellsize
        offset_z = (az-bz)/b_cellsize
        b_trimsize = aL/b_cellsize

        if offset_x<0 or offset_y<0 or offset_z<0:
            raise RuntimeError("Comparison grid %d of %r doesn't contain original grid %d of %r"%(diff_grid, diff_prefix, grid, prefix))

        try:
            offset_x,offset_y,offset_z,b_trimsize = _check_and_return_integers(offset_x,offset_y,offset_z,b_trimsize)
        except AssertionError:
            print(offset_x, offset_y, offset_z, b_trimsize, b_cellsize)
            raise RuntimeError("Comparison grid %d of %r can't be aligned to original grid %d of %r"%(diff_grid, diff_prefix, grid, prefix))

        b = b[offset_x:offset_x+b_trimsize,
              offset_y:offset_y+b_trimsize,
              offset_z:offset_z+b_trimsize]

        if len(a)>len(b):
            b = scipy.ndimage.zoom(b,len(a)//len(b),order=1)
        elif len(b)>len(a):
            b = scipy.ndimage.zoom(b,len(a)/len(b),order=1)
        a-=b

    return a

def plotslice_onegrid(prefix="output/",grid=0,slice=None,vmin=-0.15,vmax=0.15,padcells=0,offset=None,plot_offset=(0,0),
                      diff_prefix=None):
    new_plot = p.gcf().axes == []

    a = _load_grid(prefix, diff_prefix, grid)

    ax,ay,az,aL = [float(x) for x in open(prefix+"grid-info-%d.txt"%grid).readline().split()]

    if offset is None:
        offset=(-aL/2,-aL/2)
    ax+=offset[0]+plot_offset[0]
    ay+=offset[1]+plot_offset[1]

    if slice is None:
        slice = az+aL/2 + (aL/len(a))/2

    a_sl = int(len(a)*((slice-az)/aL))

    if a_sl<0 or a_sl>=len(a):
        print("Grid %d is not contained in this z-slice"%grid)
        return

    a = a[:,:,a_sl].T


    if vmin is None:
        vmin = a.real.min()
    if vmax is None:
        vmax = a.real.max()

    dx=aL/len(a)
    aL-=dx*padcells*2
    ax+=dx*padcells
    ay+=dx*padcells

    if padcells>0:
        a = a[padcells:-padcells,padcells:-padcells]

    p.imshow(a.real,extent=(ax,ax+aL,ay+aL,ay),vmin=vmin,vmax=vmax,interpolation='nearest')
    p.plot([ax,ax+aL,ax+aL,ax,ax],[ay,ay,ay+aL,ay+aL,ay],'k:')

    if new_plot:
        p.xlim(ax,ax+aL)
        p.ylim(ay,ay+aL)
    return slice, vmin, vmax, offset


def plotslice(prefix="output/",maxgrid=10,slice=None,onelevel=False,vmin=-0.15,vmax=0.15,padcells=4, offset=None,
              diff_prefix=None):
    maxgrid_on_disk = len(glob.glob(prefix+"grid-?.npy"))
    print(maxgrid_on_disk)
    if maxgrid_on_disk<maxgrid:
        maxgrid = maxgrid_on_disk

    levels = range(maxgrid)
    if onelevel:
        levels = [0]

    for level in levels:
        slice, vmin, vmax, offset = plotslice_onegrid_with_wrapping(prefix,level,slice,vmin,vmax,
                                                      padcells=0 if level==0 else padcells, offset=offset,
                                                                    diff_prefix=diff_prefix)



def plotslice_pynbody(f, slice=0.0,vmin=-0.15,vmax=0.15,use_overdensity=False):
    import pynbody
    f = pynbody.load(f)
    f.physical_units("Mpc a h^-1")
    slice/=f.properties['boxsize'].in_units('Mpc a h^-1',**f.conversion_context())
    rho_mean = f.dm['mass'].sum()/f.properties['boxsize']**3 # should be numerically equal to omegaM0
    print("rho_mean=",rho_mean)
    f.dm['delta'] = (f.dm['rho']-rho_mean)/rho_mean
    f.dm['delta'].convert_units("1")

    if use_overdensity:
        use_qty = 'overdensity'
    else:
        use_qty = 'delta'

    assert abs(f.dm['z'].max().in_units(f.properties['boxsize'])
               - f.dm['z'].min().in_units(f.properties['boxsize']) - 1.0)<0.03

    f.dm['z']-=slice

    pynbody.plot.sph.image(f.dm,qty=use_qty,width=f.properties['boxsize'],log=False,vmin=vmin,vmax=vmax,denoise=True,
                           show_cbar=False)

    return f

def plot_ps(f, with_theory=False):
    for level in range(3):
        search = f+"/*_%d.ps"%level
        ps_fname = glob.glob(search)
        if len(ps_fname)==0:
            return
        print(search,"->",ps_fname)
        k, Pk = np.loadtxt(ps_fname[0],unpack=True,usecols=(0,3))
        p.plot(k,Pk)
        if with_theory:
            k, Pk = np.loadtxt(ps_fname[0],unpack=True,usecols=(0,2))
            p.plot(k,Pk,":")
    p.loglog()
