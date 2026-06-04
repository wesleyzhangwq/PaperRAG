package com.wesz.paperrag.paper;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;

@Service
@Profile("!memory")
public class MyBatisPaperPersistenceService
    extends ServiceImpl<PaperMapper, Paper>
    implements PaperPersistenceService {
}
