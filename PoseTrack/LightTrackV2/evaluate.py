print("234")
from demo_video_mobile_original import *
from common_lib_import_and_set import *

print(123)
if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', type=str, dest='video_path',
                        default="demo/video_test.mp4")
    parser.add_argument('--model', '-m', type=str, dest='test_model',
                        default="weights/mobile-deconv/snapshot_296.ckpt")
    args = parser.parse_args()
    args.bbox_thresh = 0.4

    # initialize pose estimator
    initialize_parameters()
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    video_path = args.video_path
    visualize_folder = "demo/video_out_img"
    input_img_folder = "demo/video_input_img"
    output_video_folder = "demo/videos_out"
    output_json_folder = "demo/jsons"

    video_path_list = FileUtil.getChild_firstLevel(r"/E/dataset/PoseTrack/PoseTrack2018/images/val")
    for input_img_folder in video_path_list:
        video_name = os.path.basename(input_img_folder)
        visualize_folder = os.path.join(visualize_folder, video_name)
        output_json_path = os.path.join(output_json_folder, video_name + ".json")
        output_video_path = os.path.join(output_video_folder, video_name + "_out.mp4")

        create_folder(visualize_folder)
        create_folder(output_video_folder)
        create_folder(output_json_folder)

        light_track(pose_estimator,
                    input_img_folder, output_json_path,
                    visualize_folder, output_video_path)

        print("Finished video {}".format(output_video_path))

        ''' Display statistics '''
        # print("total_time_ALL: {:.2f}s".format(total_time_ALL))
        # print("total_time_DET: {:.2f}s".format(total_time_DET))
        # print("total_time_POSE: {:.2f}s".format(total_time_POSE))
        # print("total_time_LIGHTTRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE))
        # print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
        # print("total_num_PERSONS: {:d}\n".format(total_num_PERSONS))
        # print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
        # print("Average FPS excluding Pose Estimation: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_POSE)))
        # print("Average FPS excluding Detection: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET)))
        # print("Average FPS for framework only: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE)))
    # else:
    #     print("Video does not exist.")
