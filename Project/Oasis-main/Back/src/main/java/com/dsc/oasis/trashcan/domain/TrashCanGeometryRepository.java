package com.dsc.oasis.trashcan.domain;

import org.springframework.stereotype.Repository;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.persistence.Query;
import java.util.List;

@Repository
public class TrashCanGeometryRepository {

    @PersistenceContext
    private EntityManager entityManager;

    public List<TrashCan> getNearTCans(Double baseLatitude, Double baseLongitude){
        double distance=0.5;

        //북동쪽 좌표 구하기
        Location northEast= GeometryUtils.calculate(baseLatitude,baseLongitude,
                distance, Direction.NORTHEAST.getBearing());

        //남서쪽 좌표 구하기
        Location southWest=GeometryUtils.calculate(baseLatitude,baseLongitude,
                distance,Direction.SOUTHWEST.getBearing());

        double x1=northEast.getLatitude();
        double y1= northEast.getLongitude();
        double x2= southWest.getLatitude();
        double y2= southWest.getLongitude();

        String pointFormat=String.format("'LINESTRING(%f %f, %f %f)')",x1,y1,x2,y2);
        String setlonlat=String.format(" ST_DISTANCE_SPHERE(POINT(%f,%f), point_reverse) AS distance\n"
                ,baseLongitude,baseLatitude);

        Query query=entityManager.createNativeQuery("SELECT t.id ,t.address, t.location,t.trash_type," +
                "t.latitude, t.longitude,t.point,t.point_reverse,"+setlonlat
                +"FROM tb1_tcan AS t \n"
                + "WHERE MBRContains(ST_LINESTRINGFROMTEXT(" + pointFormat + ", t.point)", TrashCan.class)
                .setMaxResults(10);

        List<TrashCan> trashCans= query.getResultList();

        return trashCans;
    }
}
