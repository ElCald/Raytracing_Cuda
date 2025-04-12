// Includes
#include "../Geometry/geometry.h"
#include <gtest/gtest.h>
#include <cmath>

// ---- Vecteur3D tests ----

TEST(Vecteur3DTest, DotProduct)
{
    Vecteur3D a(1, 0, 0);
    Vecteur3D b(0, 1, 0);
    EXPECT_DOUBLE_EQ(a.dot(b), 0); // Orthogonal

    Vecteur3D c(2, 3, 4);
    Vecteur3D d(1, 0, -1);
    EXPECT_DOUBLE_EQ(c.dot(d), (2 * 1 + 3 * 0 + 4 * (-1))); // = -2
}

TEST(Vecteur3DTest, CrossProductOrthogonal)
{
    Vecteur3D a(1, 0, 0);
    Vecteur3D b(0, 1, 0);
    Vecteur3D result = a.cross(b);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 1); // Should be (0, 0, 1)
}

TEST(Vecteur3DTest, NormalizedLength)
{
    Vecteur3D v(3, 4, 0);
    Vecteur3D n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0, 1e-6);
}

TEST(Vecteur3DTest, Operators)
{
    Vecteur3D a(1, 2, 3);
    Vecteur3D b(4, 5, 6);
    Vecteur3D c = a + b;
    EXPECT_EQ(c.x, 5);
    EXPECT_EQ(c.y, 7);
    EXPECT_EQ(c.z, 9);

    Vecteur3D d = b - a;
    EXPECT_EQ(d.x, 3);
    EXPECT_EQ(d.y, 3);
    EXPECT_EQ(d.z, 3);

    Vecteur3D e = a * 2;
    EXPECT_EQ(e.x, 2);
    EXPECT_EQ(e.y, 4);
    EXPECT_EQ(e.z, 6);

    Vecteur3D f = b / 2;
    EXPECT_EQ(f.x, 2);
    EXPECT_EQ(f.y, 2.5);
    EXPECT_EQ(f.z, 3);
}

// ---- Point3D tests ----

TEST(Point3DTest, SubstractionGivesVector)
{
    Point3D p1(1, 2, 3);
    Point3D p2(4, 6, 9);
    Vecteur3D v = p2 - p1;

    EXPECT_EQ(v.x, 3);
    EXPECT_EQ(v.y, 4);
    EXPECT_EQ(v.z, 6);
}

TEST(Point3DTest, AddVectorToPoint)
{
    Point3D p(1, 2, 3);
    Vecteur3D v(3, 2, 1);
    Point3D result = p + v;

    EXPECT_EQ(result.x, 4);
    EXPECT_EQ(result.y, 4);
    EXPECT_EQ(result.z, 4);
}

// ---- Ray test ----

TEST(RayTest, AtFunction)
{
    Point3D origin(0, 0, 0);
    Vecteur3D direction(1, 0, 0);
    Ray ray(origin, direction);

    Point3D point = ray.at(5.0);
    EXPECT_EQ(point.x, 5);
    EXPECT_EQ(point.y, 0);
    EXPECT_EQ(point.z, 0);
}

// ---- Tests calls ----

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
