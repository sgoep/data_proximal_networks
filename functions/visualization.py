import matplotlib.patches as patches
import matplotlib.pyplot as plt

def visualization_with_zoom(x, zoom, name):
    a = 74
    b = a + 30
    c = 48
    d = c + 30

    fig, ax = plt.subplots()
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    
    if zoom:
        rect = patches.Rectangle((c, a), d-c, b-a, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        sub_axes = plt.axes([.55, .62, .25, .25]) 
        for axis in ['top','bottom','left','right']:
            sub_axes.spines[axis].set_linewidth(1.5)
        sub_axes.imshow(x[a:b, c:d], cmap='gray') 
        sub_axes.spines['bottom'].set_color('red')
        sub_axes.spines['top'].set_color('red')
        sub_axes.spines['left'].set_color('red')
        sub_axes.spines['right'].set_color('red')
        sub_axes.set_xticks([])
        sub_axes.set_yticks([])
        
    if len(name) > 0:
        plt.savefig(name, bbox_inches='tight', dpi=1200)
