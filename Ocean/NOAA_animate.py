import pygrib
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
from matplotlib import animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

fig = plt.figure()
# This program can be configured to plot and animate data on any
# collection of Grib2 files. Currently there are 12 files representing
# each month in 2018, however these files can easily be switched out
# to instead plot a different parameter and/or date range. The only
# requirement is that the location of the files does not change
directory = './Ocean/2018_global_data'

# .png generation
for filename in sorted(os.listdir(directory)):
    i = 0
    f = os.path.join(directory, filename)
    grib_file = pygrib.open(f)

    # The frequency of plotting can be controlled with the last parameter
    # in the range function below. '16' creates one drawing
    # for every two days, while '8' creates one for every day,
    # 24 would create one for every 3 days, and so on.
    g_list = grib_file.select(forecastTime=range(0, grib_file.messages, 16))

    for g in g_list:
        data, lats, lons = g.data()
        ax = plt.axes(projection=ccrs.Mercator())
        plt.pcolormesh(lons, lats, data, cmap='gist_ncar',
                       transform=ccrs.PlateCarree())
        cbar = plt.colorbar()
        cbar.set_label(g.shortName + ' (' + g.units + ')')
        ax.add_feature(cfeature.LAND)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        plt.title(g.name + '\n' + 'Valid Date: ' +
                  g.validDate.strftime('%m/%d/%Y'))
        f = './Ocean/global_{num}.png'.format(num=g.validDate.strftime('%m%d'))
        plt.savefig(f)
        fig.clear()
        if i < 10:
            print('Created drawing  {num} of {total} for month '
                  .format(num=i, total=len(g_list)) +
                  g.validDate.strftime('%m/%Y'), end='\r')
        else:
            print('Created drawing {num} of {total} for month '
                  .format(num=i, total=len(g_list)) +
                  g.validDate.strftime('%m/%Y'), end='\r')
        i += 1
    grib_file.close()

directory = './Ocean'
map_images = []
print()

# Encoded author file generation
for filename in sorted(os.listdir(directory)):
    if filename.endswith('.png'):
        f = os.path.join(directory, filename)
        img = mgimg.imread(f)
        imgplot = plt.imshow(img)
        map_images.append([imgplot])
        print('Created image plot from file ' + str(filename), end='\r')
        os.remove(f)

# .mp4 generation
my_anim = animation.ArtistAnimation(fig, map_images, blit=True)
writervideo = animation.FFMpegWriter(fps=15)
my_anim.save('global_animation.mp4', writer=writervideo)
