#ifndef CAMERA_H
#define CAMERA_H

#include "../Geometry/geometry.h"

class Camera {
    public:
        Camera();
        ~Camera() = default;


        double width;
        double height;
        double fov;
        Point3D pos;

        Vecteur3D forward;  // Direction vers l'avant
        Vecteur3D right;    // Axe horizontal
        Vecteur3D up;       // Axe vertical
};



#endif