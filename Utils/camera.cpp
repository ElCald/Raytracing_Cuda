// Includes
#include "camera.h"
#include <cmath>

// ---- Camera Implementation ----

/**
 * @param _position coordinate of the camera
 * @param _direction direction of the camera
 * @param _fov field of view of the camera
 * @param _width wifth of the camera
 * @param _height height of the camera
 */
Camera::Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height)
    : position(_position), direction(_direction.normalized()), fov(_fov), width(_width), height(_height)
{
    right = Vecteur3D(1, 0, 0);
    up = Vecteur3D(0, 1, 0);
}

/**
 * @param x coordinate x of the pixel
 * @param y coordinate y of the pixel
 * @return the generated ray
 * */
Ray Camera::generateRay(int x, int y) const
{
    double aspect_ratio = (double)width / height;
    double scale = tan(fov * 0.5 * M_PI / 180);

    double pixelNDC_x = (x + 0.5) / width;
    double pixelNDC_y = (y + 0.5) / height;

    double px = (2 * pixelNDC_x - 1) * aspect_ratio * scale;
    double py = (1 - 2 * pixelNDC_y) * scale;

    Vecteur3D ray_direction = (direction + right * px + up * py).normalized();
    return Ray(position, ray_direction);
}

/**
 * @param angle angle of the rotation
 */
void Camera::rotateX(double angle)
{
    direction.x += angle;
}

/**
 * @param angle angle of the rotation
 */
void Camera::rotateY(double angle)
{
    direction.y += angle;
}

/**
 * @param angle angle of the rotation
 * @param center center of the rotation
 */
void Camera::rotatePosX(double angle, Point3D center)
{
    position = rotateAroundX(position, center, angle);
}

/**
 * @param angle angle of the rotation
 * @param center center of the rotation
 */
void Camera::rotatePosY(double angle, Point3D center)
{
    position = rotateAroundY(position, center, angle);
}

/**
 * @param angle angle of the rotation
 * @param center center of the rotation
 */
void Camera::rotatePosZ(double angle, Point3D center)
{
    position = rotateAroundZ(position, center, angle);
}

/**
 * @param direc size of the translation
 */
void Camera::translateX(double direc)
{
    position.x += direc;
}

/**
 * @param direc size of the translation
 */
void Camera::translateY(double direc)
{
    position.y += direc;
}

/**
 * @param direc size of the translation
 */
void Camera::translateZ(double direc)
{
    position.z += direc;
}
