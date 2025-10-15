import rclpy, cv2, math, time
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PolygonStamped, Point32
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_msgs.msg import Float32
import ArducamDepthCamera as ac

cos_max_tilt = math.cos(10 * math.pi / 180)
tilt_tolerance = 10 * math.pi / 180
drone_roll = 0

def roll_callback(msg):
    #print('roll %f' % msg.data)
    global drone_roll
    drone_roll = msg.data

def swap_coordinates_filter_tilt(line):
    if line[1] > line[3]:
        line[0], line[2] = line[2], line[0]
        line[1], line[3] = line[3], line[1]
    vx = line[2] - line[0]
    vy = line[3] - line[1]
    theta = math.acos(vy / math.sqrt(vx * vx + vy * vy))
    if vx < 0:
        theta =- theta
    #print('vert line', theta)
    if -tilt_tolerance < theta - drone_roll < tilt_tolerance:
        return line
    else:
        return None

GRAD_THRESH = 300
fx = 240 / (2 * math.tan(0.5 * math.pi * 64.3 / 180));
fy = 180 / (2 * math.tan(0.5 * math.pi * 50.4 / 180));

struct_width_m = 0.1
struct_dist_m = 0.5

rclpy.init()
node = rclpy.create_node('tof')
my_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
img_pub = node.create_publisher(Image, "depth_image", my_qos)
#img_pub2 = node.create_publisher(Image, "edge_image", 1)
lines_pub = node.create_publisher(Marker, "struct_lines", my_qos)
#hori_pc_pub = node.create_publisher(PointCloud2, "hori_points", my_qos)
roll_sub = node.create_subscription(Float32, "roll", roll_callback, my_qos)
vert_hori_pub = node.create_publisher(PolygonStamped, "vert_hori_line", 1)

print("arducam sdk ver", ac.__version__)

tof = ac.ArducamCamera()
ret = 0
ret = tof.open(ac.Connection.CSI, 0)
if not ret:
    print("Failed to open camera. Error code:", ret)
    exit()
ret = tof.start(ac.FrameType.DEPTH)
if ret != 0:
    print("Failed to start camera. Error code:", ret)
    tof.close()
    exit()
tof.setControl(ac.Control.RANGE, 4)
#tof.setControl(ac.Control.FRAME_RATE, 5) # do not work
#tof.setControl(ac.Control.AUTO_FRAME_RATE, 0)
info = tof.getCameraInfo()
print(f"tof resolution: {info.width}x{info.height}")


skip_c = 0;
#kernel = np.ones((5,5),np.uint8)
print("start");

while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0)
    try:
        frame = tof.requestFrame(200)
    except KeyboardInterrupt:
        break
    if frame is not None and isinstance(frame, ac.DepthData):
        skip_c += 1
        if skip_c > 1:
            skip_c = 0

            header = Header()
            header.frame_id = "body"
            now_ns = time.monotonic_ns()
            header.stamp.sec = now_ns // 1_000_000_000
            header.stamp.nanosec = now_ns % 1_000_000_000

            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

            depth_buf[(confidence_buf < 60) | (depth_buf > 2000) | (depth_buf <= 0)] = 2000
            depth_u16 = depth_buf.astype(np.uint16)
            tof.releaseFrame(frame)

            depth_u16 = cv2.medianBlur(depth_u16, 3)
            #depth_u16 = cv2.dilate(depth_u16, kernel)

            #hist,binedge=np.histogram(depth_u16, bins=5)
            #print(f'depth image hist and bins:\n{hist}\n{binedge}')

            #edge_img = np.zeros((180, 240, 3), dtype=np.uint8)

            # detect vertical structures
            grad = cv2.Sobel(depth_u16, cv2.CV_16S, 1, 0, -1)
            ret, grad_thresh = cv2.threshold(grad, GRAD_THRESH, 255, cv2.THRESH_BINARY)
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_x_p = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 50, 5)
#            if lines_x_p is not None:
#                for line in lines_x_p:
#                    l = line[0]
#                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (128,255,255), 1, cv2.LINE_8)
            ret, grad_thresh = cv2.threshold(grad, -GRAD_THRESH, 255, cv2.THRESH_BINARY_INV);
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_x_n = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 50, 5)
#            if lines_x_n is not None:
#                for line in lines_x_n:
#                    l = line[0]
#                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (128,255,255), 1, cv2.LINE_8)
            vert_lines = None
            if lines_x_p is not None and lines_x_n is not None:
                # Precompute swapped coordinates for both lines_x_p and lines_x_n
                ok_lines_x_p = [ok_line for line in lines_x_p if (ok_line := swap_coordinates_filter_tilt(line[0])) is not None]
                ok_lines_x_n = [ok_line for line in lines_x_n if (ok_line := swap_coordinates_filter_tilt(line[0])) is not None]
                for pl in ok_lines_x_p:
                    for nl in ok_lines_x_n:
                        dx = pl[0] - nl[0]
                        dy = pl[1] - nl[1]
                        # calculate vertical struct width in px, thanks to ludovic
                        struct_width_px = struct_width_m * fy / struct_dist_m
                        # select vertical lines only if positive/negative edge separation is within a reasonable range
                        if struct_width_px - 5 <= dx <= struct_width_px + 5 and abs(dy) < 20:
                            vert_lines = (pl, nl)
                            break
                    if vert_lines is not None:
                        break

            # detect horizontal structures
            grad = cv2.Sobel(depth_u16, cv2.CV_16S, 0, 1, -1)
            ret, grad_thresh = cv2.threshold(grad, GRAD_THRESH, 255, cv2.THRESH_BINARY)
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_y = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 80, 5)
            # find the horizontal line with max length
            hori_line = None
            if lines_y is not None:
                max_len = 0
                for line in lines_y:
                    x1, y1, x2, y2 = line[0]
                    # unify direction
                    if x1 > x2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    vx = x2 - x1
                    vy = y2 - y1
                    len = math.sqrt(vx * vx + vy * vy)
                    theta = math.acos(vx / len)
                    if y2 > y1:
                        theta = -theta
                    #print('line', theta, 'roll correct', theta - drone_roll)
                    if len > max_len and -tilt_tolerance < theta - drone_roll < tilt_tolerance:
                        max_len = len
                        hori_line = (x1, y1, x2, y2)
#                    cv2.line(edge_img, (x1, y1), (x2, y2), (255,0,0), 1, cv2.LINE_8)

            line_list_points = []
            vert_hori_points = []

            if vert_lines is None:
                vert_hori_points = [Point32(), Point32()]
            else:
                pl, nl = vert_lines
                pp = np.linspace(np.array([pl[1], (pl[0]+nl[0])/2]), np.array([pl[3], (pl[2]+nl[2])/2]), num=50).astype(np.int32) # opencv y, x for numpy row, col
                ds = depth_u16[tuple(pp.T)]
                hist, bin_edges = np.histogram(ds, bins=4)
                max_i = np.argmax(hist)
                pp_3d = [(d * 0.001, (120 - p[1]) / fx * (d * 0.001), (90 - p[0]) / fy * (d * 0.001)) for p in pp if bin_edges[max_i] <= (d := depth_u16[p[0], p[1]]) <= bin_edges[max_i + 1]]

                l = cv2.fitLine(np.array(pp_3d), cv2.DIST_L2, 0, 0.01, 0.01)
                x = l[3].item(0)
                y = l[4].item(0)
                z = l[5].item(0)
                vx = l[0].item(0)
                vy = l[1].item(0)
                vz = l[2].item(0)

                if abs(vx) > 0.5 or abs(struct_dist_m - x) > 0.5:
                    # skew angle too large or vertical structure not close to horizontal structure
                    vert_hori_points = [Point32(), Point32()]
                    vert_lines = None
                else:
                    struct_dist_m = x
                    p = Point32()
                    p.x = x
                    p.y = y
                    p.z = z
                    v = Point32()
                    v.x = vx
                    v.y = vy
                    v.z = vz
                    vert_hori_points = [p, v]

                    p = Point()
                    p.x = x - vx
                    p.y = y - vy
                    p.z = z - vz
                    line_list_points.append(p)
                    p = Point()
                    p.x = x + vx
                    p.y = y + vy
                    p.z = z + vz
                    line_list_points.append(p)

            if hori_line is None:
                vert_hori_points.append(Point32())
                vert_hori_points.append(Point32())
#                if lines_y is not None:
#                    for line in lines_y:
#                        x1, y1, x2, y2 = line[0]
#                        cv2.line(edge_img, (x1, y1), (x2, y2), (255,255,255), 1, cv2.LINE_8)
            else:
                x1, y1, x2, y2 = hori_line
#                cv2.line(edge_img, (x1, y1), (x2, y2), (255,0,0), 1, cv2.LINE_8)
                if y1 >= 3 and y2 >= 3:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    n_y1 = m * 90 + b
                    n_y2 = m * 150 + b
                    pp = np.linspace(np.array([n_y1-3, 90]), np.array([n_y2-3, 150]), num=30).astype(np.int32) # opencv y, x for numpy row, col
                    pp[:,0] = np.clip(pp[:,0], 0, 180-1)
                    pp[:,1] = np.clip(pp[:,1], 0, 240-1)
                    #pp = np.linspace(np.array([y1-3, x1]), np.array([y2-3, x2]), num=50).astype(np.int32) # opencv y, x for numpy row, col
                    #ds = depth_u16[tuple(pp.T)]
                    #hist, bin_edges = np.histogram(ds, bins=4)
                    #max_i = np.argmax(hist)
                    #pp_3d = [(d * 0.001, (120 - p[1]) / fx * (d * 0.001), (90 - p[0]) / fy * (d * 0.001)) for p in pp if bin_edges[max_i] <= (d := depth_u16[p[0], p[1]]) <= bin_edges[max_i + 1]]
                    pp_3d = [(d * 0.001, (120 - p[1]) / fx * (d * 0.001), (90 - p[0]) / fy * (d * 0.001)) for p in pp for d in [depth_u16[p[0], p[1]]]]
                    #hori_pc_pub.publish(point_cloud2.create_cloud_xyz32(header, pp_3d))

                    l = cv2.fitLine(np.array(pp_3d), cv2.DIST_L2, 0, 0.01, 0.01)
                    x = l[3].item(0)
                    y = l[4].item(0)
                    z = l[5].item(0)
                    vx = l[0].item(0)
                    vy = l[1].item(0)
                    vz = l[2].item(0)
                    if abs(vx) > 0.5:
                        # skew angle too large
                        vert_hori_points.append(Point32())
                        vert_hori_points.append(Point32())
                        hori_line = None
                    else:
                        struct_dist_m = x
                        p = Point32()
                        p.x = x
                        p.y = y
                        p.z = z
                        vert_hori_points.append(p)
                        v = Point32()
                        v.x = vx
                        v.y = vy
                        v.z = vz
                        vert_hori_points.append(v)

                        p = Point()
                        p.x = x - vx
                        p.y = y - vy
                        p.z = z - vz
                        line_list_points.append(p)
                        p = Point()
                        p.x = x + vx
                        p.y = y + vy
                        p.z = z + vz
                        line_list_points.append(p)
                else:
                    vert_hori_points.append(Point32())
                    vert_hori_points.append(Point32())

            vert_hori_struct = PolygonStamped()
            vert_hori_struct.header = header
            vert_hori_struct.polygon.points = vert_hori_points
            vert_hori_pub.publish(vert_hori_struct)

            if line_list_points:
                line_list = Marker()
                line_list.header = header
                line_list.action = Marker.ADD
                line_list.type = Marker.LINE_LIST
                line_list.id = 1
                line_list.pose.orientation.w = 1.0 # 1.0, NOT 1
                line_list.scale.x = 0.05
                line_list.ns = "struct"
                line_list.color.g = 1.0
                line_list.color.a = 1.0
                line_list.lifetime.sec = 1
                line_list.points = line_list_points
                lines_pub.publish(line_list)

            if hori_line is not None:
                x1, y1, x2, y2 = hori_line
                cv2.circle(depth_u16, (x1, y1), 10, 1800, 2)
                cv2.circle(depth_u16, (x2, y2), 10, 1800, 2)
            if vert_lines is not None:
                pl, nl = vert_lines
                cv2.rectangle(depth_u16, (pl[0], pl[1]), (nl[2], nl[3]), 1800, 2)
            img = Image()
            img.header = header
            img.height = 180
            img.width = 240
            img.is_bigendian = 0
            img.encoding = "mono16"
            img.step = 240*2
            img.data = depth_u16.ravel().view(np.uint8)
            img_pub.publish(img)

#            img.header = header
#            img.encoding = "bgr8"
#            img.step = 240*3
#            img.data = edge_img.ravel().view(np.uint8)
#            img_pub2.publish(img)
        else:
            tof.releaseFrame(frame)

tof.stop()
tof.close()

rclpy.try_shutdown()

print("bye")
