from model import KMeans
from matplotlib import pyplot as plt
from utils import get_image, show_image, save_image, error


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    clusters = [2,5,10,20,50]
    #clusters = [20]
    MSE=[]
    for c in clusters:
        num_clusters = c 
        kmeans = KMeans(num_clusters)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        clustered_image = kmeans.replace_with_cluster_centers(image)

        # reshape image
        image_clustered = clustered_image.reshape(img_shape)

        # Print the error
        print('MSE for ',c,' clusters: ', error(image, clustered_image))
        MSE.append(error(image, clustered_image))
        
        # show/save image
        #show_image(image)
        save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')
    
    #print(MSE)

    fig = plt.figure(figsize = (10, 5))
    plt.bar(clusters, MSE, color ='blue',width = 1)
    plt.xlabel("No. of Clusters")
    plt.ylabel("MSE Loss")
    plt.savefig("MSE_loss")
    plt.show()


if __name__ == '__main__':
    main()
