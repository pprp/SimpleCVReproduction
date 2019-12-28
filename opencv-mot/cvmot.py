import os
import glob
import cv2

tracker_types = [
    'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE'
]


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == tracker_types[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == tracker_types[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == tracker_types[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == tracker_types[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
            print(t)

    return tracker


videoPath = "video/test.mp4"

cap = cv2.VideoCapture(videoPath)
ret, img = cap.read()
print(img.shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("out.avi", fourcc, 24, (img.shape[1], img.shape[0]))
if not cap.isOpened():
    print("Could not open video")
    exit(-1)

tracker_type = tracker_types[2]

tracker1 = createTrackerByName(tracker_type)


selection = None
track_window = None
drag_start = None


# 鼠标选框（做目标跟踪框）
def onmouse(event, x, y, flags, param):
    global selection, track_window, drag_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        track_window = None
    if drag_start:
        xmin = min(x, drag_start[0])
        ymin = min(y, drag_start[1])
        xmax = max(x, drag_start[0])
        ymax = max(y, drag_start[1])
        selection = (xmin, ymin, xmax, ymax)
    if event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        track_window = selection
        selection = None


def main():
    track_window1 = ()
    cv2.namedWindow('image', 1)
    cv2.setMouseCallback('image', onmouse)
    # We will track the frames as we load them off of disk
    # for k, f in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
    k = 0
    while (1):
        ret, frame = cap.read()
        if not ret:
            print("Game over!")
            break
        print("Processing Frame {}".format(k))
        img_raw = frame  #cv2.imread(f)
        image = img_raw.copy()

        # We need to initialize the tracker on the first frame
        if k == 0:
            # Start a track on the object you want. box the object using the mouse and press 'Enter' to start tracking
            while True:
                ret, image = cap.read()
                img_first = image.copy()
                if track_window:
                    cv2.rectangle(img_first,
                                  (track_window[0], track_window[1]),
                                  (track_window[2], track_window[3]),
                                  (0, 0, 255), 1)
                elif selection:
                    cv2.rectangle(img_first, (selection[0], selection[1]),
                                  (selection[2], selection[3]), (0, 0, 255), 1)
                if track_window1:
                    cv2.rectangle(img_first,
                                  (track_window1[0], track_window1[1]),
                                  (track_window1[2], track_window1[3]),
                                  (0, 255, 255), 1)

                cv2.imshow('image', img_first)
                if cv2.waitKey(10) == 32:  # space
                    track_window1 = list(track_window)
                    track_window1[2] = track_window1[2] - track_window1[0]
                    track_window1[3] = track_window1[3] - track_window1[1]
                    track_window1 = tuple(track_window1)
                    k = 2
                    tracker1.init(image, track_window1)
                    break

        else:
            # Else we just attempt to track from the previous frame
            status, box1_predict = tracker1.update(image)

            # Get previous box and draw on showing image
            cv2.rectangle(image, (int(box1_predict[0]), int(box1_predict[1])),
                          (int(box1_predict[0] + box1_predict[2]),
                           int(box1_predict[1] + box1_predict[3])),
                          (0, 255, 255), 3)
            cv2.putText(image, tracker_type, (150, 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
            #cv2.putText(image, "standard", (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('image', image)
            out.write(image)
            # cv2.waitKey(10)
            c = cv2.waitKey(10) & 0xff
            if c == 27: break  # ESC
            k += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()