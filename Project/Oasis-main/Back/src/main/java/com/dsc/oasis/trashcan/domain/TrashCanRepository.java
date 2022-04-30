package com.dsc.oasis.trashcan.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface TrashCanRepository extends JpaRepository<TrashCan,Long> {
    Optional<TrashCan> findById(Long id);
}
