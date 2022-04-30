package com.dsc.oasis.trashcan.domain;

        /**
         * Calculate destination with start point, distance, bearing
         *
        * φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
        * λ2 = λ1 + atan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )
        *
        * φ is lat, λ is lng, θ is the bearing (clockwise from N),
        * δ is the angular dist d/R; d being the dist travelled, R the earth’s radius
        *
        * param baseLatitude =latitude of start point
        * param startLongi =longitude of start point
        * param distance =distance to calculate, unit is km (ex, 1.0 will 1km)
        * param bearing =bearing to calculate, range is 0 to 360.
        **/

public class GeometryUtils {
    public static Location calculate(Double baseLatitude, Double baseLongitude, Double distance,
                                     Double bearing) {
        Double radianLatitude = toRadian(baseLatitude);
        Double radianLongitude = toRadian(baseLongitude);
        Double radianAngle = toRadian(bearing);
        Double distanceRadius = distance / 6371.01;
        Double latitude = Math.asin(sin(radianLatitude) * cos(distanceRadius) +
                cos(radianLatitude) * sin(distanceRadius) * cos(radianAngle));
        Double longitude = radianLongitude + Math.atan2(sin(radianAngle) * sin(distanceRadius) *
                cos(radianLatitude), cos(distanceRadius) - sin(radianLatitude) * sin(latitude));
        longitude = normalizeLongitude(longitude);
        return new Location(toDegree(latitude), toDegree(longitude));
    }
    private static Double toRadian(Double coordinate) {
        return coordinate * Math.PI / 180.0;
    }
    private static Double toDegree(Double coordinate) {
        return coordinate * 180.0 / Math.PI;
    }
    private static Double sin(Double coordinate) {
        return Math.sin(coordinate);
    }
    private static Double cos(Double coordinate) {
        return Math.cos(coordinate);
    }
    private static Double normalizeLongitude(Double longitude) {
        return (longitude + 540) % 360 - 180;
    }
}
