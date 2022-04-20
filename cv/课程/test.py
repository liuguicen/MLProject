## Annotations

In the PoseTrack benchmark each person is labeled with a head bounding box and positions of the body joints. We omit annotations of people in dense crowds and in some cases also choose to skip annotating people in upright standing poses. This is done to focus annotation efforts on the relevant people in the scene. We include ignore regions to specify which people in the image where ignored during annotation.

Each sequence included in the PoseTrack benchmark correspond to about 5 seconds of video. The number of frames in each sequence might vary as different videos were recorded with different number of frames per second. For the **training** sequences we provide annotations for 30 consecutive frames centered in the middle of the sequence. For the **validation and test ** sequences we annotate 30 consecutive frames and in addition annotate every 4-th frame of the sequence. The rationale for that is to evaluate both smoothness of the estimated body trajectories as well as ability to generate consistent tracks over longer temporal span. Note, that even though we do not label every frame in the provided sequences we still expect the unlabeled frames to be useful for achieving better performance on the labeled frames.

## Annotation Format

The PoseTrack 2018 submission file format is based on the Microsoft COCO dataset annotation format. We decided for this step to 1) maintain compatibility to a commonly used format and commonly used tools while 2) allowing for sufficient flexibility for the different challenges. These are the 2D tracking challenge, the 3D tracking challenge as well as the dense 2D tracking challenge.

Furthermore, we require submissions in a zipped version of either one big .json file or one .json file per sequence to 1) be flexible w.r.t. tools for each sequence (e.g., easy visualization for a single sequence independent of others and 2) to avoid problems with file size and processing.

The MS COCO file format is a nested structure of dictionaries and lists.  For evaluation, we only need a subsetof the standard fields, however a few  additional fields are required for the evaluation protocol (e.g., a confidence  value for every estimated body landmark). In the following we describe the minimal,  but required set of fields for a submission. Additional fields may be present,  but are ignored by the evaluation script.

**Overall, the submission file must be a .zip file of either one .json file for all test sequences or one .json file per test sequence.**

### .json dictionary structure

At top level, each .json file stores a dictionary with three elements:

* images
* annotations
* categories

#### The ‘images’ element is a list of described images in this file. The list must contain the information  for all images referenced by a person description in the file. Each list element is a dictionary  and must contain only two fields: `file_name` and `id` (unique int). The file name must refer to the original posetrack image as extracted from the test set, e.g., `images/test/023736_mpii_test/000000.jpg`.

#### The ‘annotations’ element
 This is another list of dictionaries. Each item of the list describes one detected person and is itself a dictionary. It must have at least the following fields:

* `image_id` (int, an image with a corresponding id must be in `images`),
* `track_id` (int, the track this person is performing; unique per frame),`
* `keypoints` (list of floats, length three times number of estimated keypoints   in order x, y, ? for every point. The third value per keypoint is only there   for COCO format consistency and not used.),
* `scores` (list of float, length number of estimated keypoints; each value   between 0. and 1. providing a prediction confidence for each keypoint),

#### The ‘categories’ element
 The categories element must be a list containing precisely one item, describing the person structure. The dictionary must contain a field `name`: `person` as well as a field `keypoints`. The keypoints field is a list of strings which must be a superset of [`nose`, `upper_neck`, `head_top`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`]. The order may be arbitrary.

## Submission Policy
Please register at the dataset [webpage](www.posetrack.net) to upload your predictions. Note that multiple registrations are not allowed. Users can evaluate their approach on either validation set or held-out test set. For held-out test set, at most **four** submissions per task can be made for the same approach. Any two submissions should be **72 hours** apart. We advise you to evaluate your approach on validation data first to avoid losing your evaluation attempts due to formatting mismatch or other problems. Evaluation on validation set has no submission limit. Ablation studies of your approach can be performed on the validation set.

