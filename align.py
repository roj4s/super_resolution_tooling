import cv2
import numpy as np
import os
import argparse
from dataset_iterator import SRDatasetIterator


def align_sift(align_img_addr, ref_img_addr, output_addr=None, min_match_count=4):
    align_img = cv2.imread(align_img_addr)
    ref_img = cv2.imread(ref_img_addr)
    align_gray = cv2.cvtColor(align_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

    align_kpts, align_descripts = sift.detectAndCompute(align_gray,None)
    ref_kpts, ref_descripts = sift.detectAndCompute(ref_gray,None)

    matches = matcher.knnMatch(ref_descripts, align_descripts, 2)
    matches = sorted(matches, key = lambda x:x[0].distance)

    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    canvas = align_img.copy()

    if len(good) > min_match_count:
        src_pts = np.float32([ ref_kpts[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ align_kpts[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        h,w = ref_img.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    else:
        print(f"Not enough matches found - {len(food)}/{min_match_count}")

    matched = cv2.drawMatches(ref_img, ref_kpts, canvas, align_kpts, good, None)

    h,w = ref_img.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
    found = cv2.warpPerspective(align_img, perspectiveM, (w,h))

    if output_addr is not None:
        img_name, img_format = os.path.basename(align_img_addr).split('.')
        output_addr_matches = os.path.join(output_addr,
                                   f"sift_matches_{img_name}.{img_format}")
        output_addr_align = os.path.join(output_addr,
                                   f"{img_name}.{img_format}")

        cv2.imwrite(output_addr_matches, matched)
        cv2.imwrite(output_addr_align, found)

    return found, matched


def align_sift_dataset_iterator(dataset_iterator: SRDatasetIterator,
                                output_addr=None):
    for img_data in dataset_iterator.get_pair():
        yield align_sift(img_data['lr'], img_data['hr'], output_addr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay images for alignment' \
                                     'visual check')
    parser.add_argument('align_img_addr', type=str, help='Image to align')
    parser.add_argument('reference_img_addr', type=str, help='Image to use as'\
                        ' reference')
    parser.add_argument('--isdir', help='From a dataset', action='store_true', default=False)
    parser.add_argument('--output', help='Optional directory to output results', type=str,
                        default=None)
    parser.add_argument('--show', help='Show output', action='store_true', default=False)

    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    if args.output is None:
        args.show = True

    imgs = []
    if args.isdir:
        it = SRDatasetIterator(args.reference_img_addr)
        imgs = [(aligned, matches) for (aligned, matches) in
                align_sift_dataset_iterator(it, args.output)]
    else:
        imgs.append(align_sift(args.align_img_addr, args.reference_img_addr, args.output))

    if args.show:
        for aligned, matches in imgs:
            cv2.imshow('aligned', aligned)
            cv2.imshow('matches', matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()