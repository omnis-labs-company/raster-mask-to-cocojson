#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <sstream>
#include <thread>
#include <mutex>

#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/filewritestream.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/writer.h"
#include "rapidjson/include/rapidjson/prettywriter.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace rapidjson;

class Annotation {
public:
    Annotation(int categoryId, int imageId, const vector<Point>& polygonPoints)
        : category_id(categoryId), image_id(imageId), polygon(polygonPoints) {}

    int getCategoryId() const { return category_id; }
    int getImageId() const { return image_id; }
    const vector<Point>& getPolygon() const { return polygon; }

private:
    int category_id;
    int image_id;
    vector<Point> polygon;
};

string vec3bToString(const Vec3b& color) {
    return to_string(color[0]) + "," + to_string(color[1]) + "," + to_string(color[2]);
}

// Process a specific color area and extract contours
void processColorRegion(const Mat& image, const Vec3b color, int classId, int imageId, vector<Annotation>& annotations) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.at<Vec3b>(y, x) == color) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        if (contour.size() >= 3) {
            annotations.emplace_back(classId, imageId, contour);
        }
    }
}

// Process one image and accumulate annotation results
void processImage(const string& maskPath,
                  const map<string, string>& colorToLabel,
                  const map<string, int>& labelToId,
                  int imageId,
                  vector<Annotation>& annotations) {
    Mat maskImage = imread(maskPath);
    if (maskImage.empty()) return;

    for (const auto& [colorStr, label] : colorToLabel) {
        Vec3b color;
        sscanf(colorStr.c_str(), "%hhu,%hhu,%hhu", &color[2], &color[1], &color[0]); // Compare the channel values using OpenCV's BGR format

        int classId = labelToId.at(label);
        processColorRegion(maskImage, color, classId, imageId, annotations);
    }
}

// Save all annotations to a COCO-format JSON file
void writeCocoJson(
    const vector<Annotation>& annotations,
    const vector<string>& imageNames,
    const string& outputPath,
    const map<string, int>& labelToId
) {
    Document doc;
    doc.SetObject();
    Document::AllocatorType& allocator = doc.GetAllocator();

    Value images(kArrayType);
    Value anns(kArrayType);
    Value categories(kArrayType);

    int annId = 1;
    for (int i = 0; i < imageNames.size(); ++i) {
        Value image(kObjectType);
        image.AddMember("id", i + 1, allocator);
        image.AddMember("file_name", StringRef(imageNames[i].c_str()), allocator);
        image.AddMember("width", 1664, allocator);
        image.AddMember("height", 832, allocator);
        images.PushBack(image, allocator);
    }

    for (const auto& ann : annotations) {
        Value annotation(kObjectType);
        annotation.AddMember("id", annId++, allocator);
        annotation.AddMember("image_id", ann.getImageId(), allocator);
        annotation.AddMember("category_id", ann.getCategoryId(), allocator);
        annotation.AddMember("iscrowd", 0, allocator);

        Value segmentation(kArrayType);
        Value polygon(kArrayType);
        for (const Point& pt : ann.getPolygon()) {
            polygon.PushBack(pt.x, allocator);
            polygon.PushBack(pt.y, allocator);
        }
        segmentation.PushBack(polygon, allocator);
        annotation.AddMember("segmentation", segmentation, allocator);

        Rect bbox = boundingRect(ann.getPolygon());
        Value bboxVal(kArrayType);
        bboxVal.PushBack(bbox.x, allocator);
        bboxVal.PushBack(bbox.y, allocator);
        bboxVal.PushBack(bbox.width, allocator);
        bboxVal.PushBack(bbox.height, allocator);
        annotation.AddMember("bbox", bboxVal, allocator);
        annotation.AddMember("area", bbox.width * bbox.height, allocator);

        anns.PushBack(annotation, allocator);
    }

    for (const auto& [label, id] : labelToId) {
        Value cat(kObjectType);
        cat.AddMember("id", id, allocator);
        cat.AddMember("name", StringRef(label.c_str()), allocator);
        cat.AddMember("supercategory", "none", allocator);
        categories.PushBack(cat, allocator);
    }

    doc.AddMember("images", images, allocator);
    doc.AddMember("annotations", anns, allocator);
    doc.AddMember("categories", categories, allocator);

    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    doc.Accept(writer);

    ofstream outFile(outputPath);
    outFile << buffer.GetString();
    outFile.close();
}

int main() {
    string maskFolder = "CVRG-Pano-20250314T184415Z-001/CVRG-Pano/all-rgb-masks";
    string outputJson = "output.json";

    map<string, string> colorToLabel = {
        {"0,0,0", "background"}, {"113,174,206", "car"}, {"0,64,64", "bicycle"}, {"224,128,0", "bridge"},
        {"240,240,20", "building"}, {"32,64,128", "bus"}, {"32,224,128", "fence"}, {"64,0,160", "ground"},
        {"128,160,160", "motorcycle"}, {"230,63,228", "parking area"}, {"192,96,96", "person"}, {"255,0,124", "pole"},
        {"250,125,187", "road"}, {"240,146,26", "sidewalk"}, {"64,117,128", "sky"}, {"240,64,64", "terrain"},
        {"19,92,211", "traffic light"}, {"236,28,159", "traffic sign"}, {"144,96,128", "truck"},
        {"144,32,192", "vegetation"}, {"208,32,192", "wall"}
    };

    map<string, int> labelToId;
    int classCounter = 1;
    for (const auto& [_, label] : colorToLabel) {
        if (labelToId.count(label) == 0) {
            labelToId[label] = classCounter++;
        }
    }

    vector<string> maskFiles;
    for (const auto& entry : fs::directory_iterator(maskFolder)) {
        maskFiles.push_back(entry.path().filename().string());
    }

    vector<Annotation> allAnnotations;
    vector<vector<Annotation>> threadAnnotations(4); // 4 threads
    vector<thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = t; i < maskFiles.size(); i += 4) {
                string maskPath = maskFolder + "/" + maskFiles[i];
                processImage(maskPath, colorToLabel, labelToId, i + 1, threadAnnotations[t]);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    for (const auto& vec : threadAnnotations) {
        allAnnotations.insert(allAnnotations.end(), vec.begin(), vec.end());
    }

    writeCocoJson(allAnnotations, maskFiles, outputJson, labelToId);

    return 0;
}