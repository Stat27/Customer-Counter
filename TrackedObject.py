class TrackedObject:
    # to keep track of the last used ID
    _id_counter = 0  

    def __init__(self, centroid):
        self.centroid = centroid
        self.path = [centroid]
        self.id = TrackedObject._id_counter
        self.counted = False
        # Increment ID for next object
        TrackedObject._id_counter += 1  
    
    # update the centroid of this tracked object
    def update(self, new_centroid):
        self.centroid = new_centroid
        self.path.append(new_centroid)