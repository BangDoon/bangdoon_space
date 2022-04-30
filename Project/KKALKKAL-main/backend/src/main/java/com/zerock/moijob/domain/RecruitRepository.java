package com.zerock.moijob.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RecruitRepository extends JpaRepository<Recruit,Long> {
    List<Recruit> findAllByRegionContaining(String region);
}
